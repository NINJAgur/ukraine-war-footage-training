terraform {
  required_providers {
    google = { source = "hashicorp/google", version = "~> 5.0" }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_storage_bucket" "media" {
  name          = var.bucket_name
  location      = upper(var.region)
  force_destroy = false
  uniform_bucket_level_access = true

  cors {
    origin          = ["*"]
    method          = ["GET"]
    response_header = ["Content-Type"]
    max_age_seconds = 3600
  }
}

resource "google_storage_bucket_iam_member" "public_read" {
  bucket = google_storage_bucket.media.name
  role   = "roles/storage.objectViewer"
  member = "allUsers"
}


data "google_project" "project" {}

resource "google_storage_bucket_iam_member" "compute_rw" {
  bucket = google_storage_bucket.media.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${data.google_project.project.number}-compute@developer.gserviceaccount.com"
}


# ── e2-micro (CPU services) ───────────────────────────────────────────

resource "google_compute_address" "e2_micro" {
  name   = "ukraine-footage-main-ip"
  region = var.region
}

resource "google_compute_firewall" "allow_web" {
  name    = "allow-web"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["80", "443", "22"]
  }
  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["web"]
}

resource "google_compute_firewall" "allow_internal" {
  name    = "allow-internal-services"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["5432", "6379", "8000"]
  }
  # T4 VM internal IP range (same VPC)
  source_ranges = ["10.128.0.0/9"]
  target_tags   = ["web"]
}

resource "google_compute_instance" "e2_micro" {
  name         = "ukraine-footage-main"
  machine_type = "e2-micro"
  zone         = var.zone
  tags         = ["web"]

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 30
      type  = "pd-standard"
    }
  }

  network_interface {
    network = "default"
    access_config {
      nat_ip = google_compute_address.e2_micro.address
    }
  }

  metadata = {
    "ssh-keys"       = var.ssh_deploy_key_pub != "" ? "ubuntu:${var.ssh_deploy_key_pub}" : null
    "startup-script" = <<-EOF
      #!/bin/bash
      set -e
      exec > /var/log/startup-script.log 2>&1

      # Swap — prevents OOM during Docker builds on e2-micro
      if [ ! -f /swapfile ]; then
        fallocate -l 2G /swapfile
        chmod 600 /swapfile
        mkswap /swapfile
        swapon /swapfile
        echo '/swapfile none swap sw 0 0' >> /etc/fstab
      fi

      # Docker + git
      apt-get update -qq
      apt-get install -y -qq docker.io docker-compose-v2 git
      systemctl enable docker
      systemctl start docker

      # Clone repo (once)
      if [ ! -d /home/ubuntu/app ]; then
        git clone --depth=1 https://github.com/NINJAgur/ukraine-war-footage-training.git /home/ubuntu/app
        chown -R ubuntu:ubuntu /home/ubuntu/app
      fi

      # Write .env (refreshed every boot)
      cat > /home/ubuntu/app/.env <<ENVEOF
POSTGRES_PASSWORD=${var.postgres_password}
CORS_ORIGINS=${var.cors_origins}
STORAGE_MODE=remote
REMOTE_STORAGE_BUCKET=${var.bucket_name}
JWT_SECRET=${var.jwt_secret}
ADMIN_USERNAME=${var.admin_username}
ADMIN_PASSWORD=${var.admin_password}
ENVEOF
      chown ubuntu:ubuntu /home/ubuntu/app/.env

      # Build and start all CPU services
      cd /home/ubuntu/app
      sudo -u ubuntu docker compose -f docker-compose.prod.yml pull 2>/dev/null || true
      docker compose -f docker-compose.prod.yml up -d --build
    EOF
  }

  service_account {
    scopes = ["cloud-platform"]
  }
}

output "e2_micro_ip" {
  value = google_compute_address.e2_micro.address
}

# ── Persistent datasets disk (Kaggle data + merged dirs) ─────────────

resource "google_compute_disk" "datasets" {
  name = "ukraine-footage-datasets"
  zone = var.zone
  type = "pd-standard"
  size = 150

  lifecycle {
    prevent_destroy = true
  }
}

# ── inference-engine VM (CPU, daily 03:00–04:00 UTC) ─────────────────
# 1-hour window: GDINO @03:05 + annotate @03:35; stops at 04:00 (30-min buffer before training starts)

resource "google_compute_resource_policy" "inference_schedule" {
  name   = "inference-daily-schedule"
  region = var.region

  instance_schedule_policy {
    vm_start_schedule { schedule = "0 3 * * *" }
    vm_stop_schedule  { schedule = "0 4 * * *" }
    time_zone         = "UTC"
  }
}

# ── training-engine Instance Schedule (04:30 UTC start, self-shuts) ──
# Standard (non-preemptible) required — Spot VMs cannot have Instance Scheduling policies.
# Self-shutdown: startup script exits early if no QUEUED runs; train_finetune shuts down after last model.

resource "google_compute_resource_policy" "training_schedule" {
  name   = "training-daily-schedule"
  region = var.region

  instance_schedule_policy {
    vm_start_schedule { schedule = "30 4 * * *" }
    time_zone         = "UTC"
  }
}

resource "google_compute_instance" "inference_engine" {
  name                      = "ukraine-footage-inference"
  machine_type              = "n1-standard-1"
  zone                      = var.inference_zone
  tags                      = ["inference-worker"]
  allow_stopping_for_update = true

  scheduling {
    provisioning_model          = "SPOT"
    instance_termination_action = "STOP"
    preemptible                 = true
    automatic_restart           = false
    on_host_maintenance         = "TERMINATE"
  }

  guest_accelerator {
    type  = "nvidia-tesla-t4"
    count = 1
  }

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 50
      type  = "pd-standard"
    }
  }

  network_interface {
    network = "default"
    access_config {}
  }

  resource_policies = [google_compute_resource_policy.inference_schedule.id]

  metadata = {
    "ssh-keys"       = var.ssh_deploy_key_pub != "" ? "ubuntu:${var.ssh_deploy_key_pub}" : null
    "startup-script" = <<-EOF
      #!/bin/bash
      set -e
      exec >> /var/log/startup-script.log 2>&1

      # First-boot: install system deps + NVIDIA driver
      if [ ! -f /var/lib/nvidia-driver-installed ]; then
        apt-get update -qq
        apt-get install -y --no-install-recommends \
          build-essential gcc-12 linux-headers-$(uname -r) \
          ubuntu-drivers-common git python3 python3-venv python3-pip wget ffmpeg
        update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 12
        ubuntu-drivers autoinstall
        touch /var/lib/nvidia-driver-installed
        reboot
        exit 0
      fi

      # Wait for NVIDIA driver
      until nvidia-smi &>/dev/null; do sleep 10; done

      # Sparse clone on first boot only
      if [ ! -d /home/ubuntu/app ]; then
        git clone --depth=1 --filter=blob:none --sparse \
          https://github.com/NINJAgur/ukraine-war-footage-training.git \
          /home/ubuntu/app
        cd /home/ubuntu/app
        git sparse-checkout set inference-engine shared
        git sparse-checkout reapply
        chown -R ubuntu:ubuntu /home/ubuntu/app
      fi

      # Python venv + CUDA torch + deps (skip if already built)
      cd /home/ubuntu/app/inference-engine
      if [ ! -f venv/bin/celery ]; then
        python3 -m venv venv
        venv/bin/pip install --quiet "torch==2.5.1+cu121" "torchvision==0.20.1+cu121" \
          --index-url https://download.pytorch.org/whl/cu121
        grep -v "^torch\|^torchvision" requirements.txt > /tmp/requirements_gpu.txt
        venv/bin/pip install --quiet -r /tmp/requirements_gpu.txt
      fi

      # GroundingDINO checkpoint (661MB, gitignored — download once)
      if [ ! -f /home/ubuntu/app/inference-engine/groundingdino_swint_ogc.pth ]; then
        wget -q -O /home/ubuntu/app/inference-engine/groundingdino_swint_ogc.pth \
          https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
      fi

      # Download weights from GCS on first boot
      if [ ! -d /home/ubuntu/app/training-engine/runs ]; then
        venv/bin/python3 - <<PYEOF
from google.cloud import storage
import pathlib
client = storage.Client()
training_root = pathlib.Path("/home/ubuntu/app/training-engine")
for blob in client.list_blobs("${var.bucket_name}", prefix="runs/"):
    if not blob.name.endswith("best.pt"):
        continue
    local = training_root / blob.name
    local.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local))
    print(f"Downloaded {blob.name}")
PYEOF
      fi

      # Write .env (refreshed every boot in case IPs change)
      cat > /home/ubuntu/app/.env <<ENVEOF
DATABASE_SYNC_URL=postgresql+psycopg2://postgres:${var.postgres_password}@${google_compute_instance.e2_micro.network_interface[0].network_ip}:5432/ukraine_footage
CELERY_BROKER_URL=redis://${google_compute_instance.e2_micro.network_interface[0].network_ip}:6379/0
CELERY_RESULT_BACKEND=redis://${google_compute_instance.e2_micro.network_interface[0].network_ip}:6379/1
REDIS_URL=redis://${google_compute_instance.e2_micro.network_interface[0].network_ip}:6379/0
STORAGE_MODE=remote
REMOTE_STORAGE_BUCKET=${var.bucket_name}
GPU_DEVICE=cuda:0
ENVEOF
      chown ubuntu:ubuntu /home/ubuntu/app/.env

      # systemd service
      cat > /etc/systemd/system/celery-inference.service <<SVCEOF
[Unit]
Description=Celery Inference Worker + Beat
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/app/inference-engine
EnvironmentFile=/home/ubuntu/app/.env
Environment=OMP_NUM_THREADS=4
ExecStart=/home/ubuntu/app/inference-engine/venv/bin/celery -A celery_app worker -Q pipeline --pool=prefork --concurrency=1 --loglevel=info --beat
Restart=on-failure
RestartSec=30
StandardOutput=append:/var/log/celery-inference.log
StandardError=append:/var/log/celery-inference.log

[Install]
WantedBy=multi-user.target
SVCEOF

      systemctl daemon-reload
      systemctl enable celery-inference
      systemctl start celery-inference
    EOF
  }

  service_account {
    scopes = ["cloud-platform"]
  }
}

output "inference_engine_ip" {
  value = google_compute_instance.inference_engine.network_interface[0].access_config[0].nat_ip
}

# ── training-engine VM (n1-standard-4 + T4, Instance Schedule 04:30 UTC start) ─
# Spot VM. Self-shuts after training or immediately if no QUEUED runs.

resource "google_compute_instance" "training_engine" {
  name                      = "ukraine-footage-training"
  machine_type              = "n1-standard-4"
  zone                      = var.zone
  tags                      = ["training-worker"]
  allow_stopping_for_update = true

  scheduling {
    provisioning_model          = "SPOT"
    instance_termination_action = "STOP"
    preemptible                 = true
    automatic_restart           = false
    on_host_maintenance         = "TERMINATE"
  }

  resource_policies = [google_compute_resource_policy.training_schedule.id]

  guest_accelerator {
    type  = "nvidia-tesla-t4"
    count = 1
  }

  boot_disk {
    initialize_params {
      image = "ubuntu-os-cloud/ubuntu-2204-lts"
      size  = 50
      type  = "pd-standard"
    }
  }

  attached_disk {
    source      = google_compute_disk.datasets.self_link
    device_name = "datasets"
  }

  network_interface {
    network = "default"
    access_config {}
  }

  metadata = {
    "ssh-keys"     = var.ssh_deploy_key_pub != "" ? "ubuntu:${var.ssh_deploy_key_pub}" : null
    startup-script = <<-EOF
      #!/bin/bash
      set -e
      exec >> /var/log/startup-script.log 2>&1

      # Install NVIDIA driver on first boot, then reboot to load kernel module
      if [ ! -f /var/lib/nvidia-driver-installed ]; then
        apt-get update -qq
        apt-get install -y --no-install-recommends \
          build-essential gcc-12 linux-headers-$(uname -r) \
          ubuntu-drivers-common git python3 python3-venv python3-pip wget ffmpeg
        update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 12
        ubuntu-drivers autoinstall
        touch /var/lib/nvidia-driver-installed
        reboot
        exit 0
      fi

      # Wait for NVIDIA driver
      until nvidia-smi &>/dev/null; do sleep 10; done

      # Sparse clone on first boot only
      if [ ! -d /home/ubuntu/app ]; then
        git clone --depth=1 --filter=blob:none --sparse \
          https://github.com/NINJAgur/ukraine-war-footage-training.git \
          /home/ubuntu/app
        cd /home/ubuntu/app
        git sparse-checkout set training-engine shared
        git sparse-checkout reapply
        chown -R ubuntu:ubuntu /home/ubuntu/app
      fi

      # Python venv + deps
      cd /home/ubuntu/app/training-engine
      if [ ! -f venv/bin/celery ]; then
        python3 -m venv venv
        venv/bin/pip install -r requirements.txt
      fi

      # Download weights from GCS (needed as baseline_weights for train_finetune)
      if [ ! -d /home/ubuntu/app/training-engine/runs ]; then
        pip3 install --quiet google-cloud-storage
        python3 - <<PYEOF
from google.cloud import storage
import pathlib
client = storage.Client()
training_root = pathlib.Path("/home/ubuntu/app/training-engine")
for blob in client.list_blobs("ukraine-footage-media", prefix="runs/"):
    if not blob.name.endswith("best.pt"):
        continue
    local = training_root / blob.name
    local.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local))
    print(f"Downloaded {blob.name}")
PYEOF
      fi

      # Mount persistent datasets disk (Kaggle cache)
      DATASETS_DEV="/dev/disk/by-id/google-datasets"
      DATASETS_MNT="/mnt/datasets"
      mkdir -p "$DATASETS_MNT"
      if ! blkid "$DATASETS_DEV" | grep -q ext4; then
        mkfs.ext4 -F "$DATASETS_DEV"
      fi
      mount -o discard,defaults "$DATASETS_DEV" "$DATASETS_MNT" || true
      resize2fs "$DATASETS_DEV" 2>/dev/null || true
      grep -q "$DATASETS_MNT" /etc/fstab || \
        echo "$DATASETS_DEV $DATASETS_MNT ext4 discard,defaults,nofail 0 2" >> /etc/fstab

      # Swap on persistent disk
      if [ ! -f "$DATASETS_MNT/swapfile" ]; then
        fallocate -l 4G "$DATASETS_MNT/swapfile"
        chmod 600 "$DATASETS_MNT/swapfile"
        mkswap "$DATASETS_MNT/swapfile"
      fi
      swapon "$DATASETS_MNT/swapfile" 2>/dev/null || true

      # Redirect kagglehub cache to persistent disk
      mkdir -p "$DATASETS_MNT/.cache/kagglehub"
      mkdir -p /home/ubuntu/.cache
      chown -R ubuntu:ubuntu "$DATASETS_MNT/.cache" /home/ubuntu/.cache
      ln -sfn "$DATASETS_MNT/.cache/kagglehub" /home/ubuntu/.cache/kagglehub

      # Symlink training-engine/media → persistent disk
      mkdir -p "$DATASETS_MNT/media"
      chown -R ubuntu:ubuntu "$DATASETS_MNT"
      if [ ! -L /home/ubuntu/app/training-engine/media ]; then
        rm -rf /home/ubuntu/app/training-engine/media
        ln -s "$DATASETS_MNT/media" /home/ubuntu/app/training-engine/media
      fi

      # Download Kaggle datasets + build merged folders (once — persists on disk)
      cd /home/ubuntu/app/training-engine
      sudo -u ubuntu bash scripts/setup_datasets.sh || \
        echo "[startup] Dataset setup incomplete — will retry on next boot"

      # Write .env
      cat > /home/ubuntu/app/.env <<ENVEOF
DATABASE_SYNC_URL=postgresql+psycopg2://postgres:${var.postgres_password}@${google_compute_instance.e2_micro.network_interface[0].network_ip}:5432/ukraine_footage
CELERY_BROKER_URL=redis://${google_compute_instance.e2_micro.network_interface[0].network_ip}:6379/0
CELERY_RESULT_BACKEND=redis://${google_compute_instance.e2_micro.network_interface[0].network_ip}:6379/1
REDIS_URL=redis://${google_compute_instance.e2_micro.network_interface[0].network_ip}:6379/0
STORAGE_MODE=remote
REMOTE_STORAGE_BUCKET=ukraine-footage-media
ENVEOF
      chown ubuntu:ubuntu /home/ubuntu/app/.env

      # Early exit: shut down immediately if no QUEUED training runs
      QUEUED_COUNT=$(su - ubuntu -c "cd /home/ubuntu/app/training-engine && \
        venv/bin/python3 -c \"
import os, sys
sys.path.insert(0, 'shared')
os.environ.setdefault('DATABASE_SYNC_URL', open('/home/ubuntu/app/.env').read().split('DATABASE_SYNC_URL=')[1].split('\n')[0])
from sqlalchemy import create_engine, text
engine = create_engine(os.environ['DATABASE_SYNC_URL'])
with engine.connect() as conn:
    result = conn.execute(text(\\\"SELECT COUNT(*) FROM training_runs WHERE status='QUEUED'\\\"))
    print(result.scalar())
\"" 2>/dev/null || echo "0")
      if [ "$QUEUED_COUNT" = "0" ]; then
        echo "[startup] No QUEUED training runs — shutting down." >> /var/log/startup-script.log
        sudo shutdown -h now
        exit 0
      fi
      echo "[startup] Found $QUEUED_COUNT QUEUED training run(s) — starting worker." >> /var/log/startup-script.log

      # systemd service
      cat > /etc/systemd/system/celery-training.service <<SVCEOF
[Unit]
Description=Celery Training Worker
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/app/training-engine
EnvironmentFile=/home/ubuntu/app/.env
ExecStart=/home/ubuntu/app/training-engine/venv/bin/celery -A celery_app worker -Q training --pool=solo --concurrency=1 --loglevel=info
Restart=on-failure
RestartSec=30
StandardOutput=append:/var/log/celery-training.log
StandardError=append:/var/log/celery-training.log

[Install]
WantedBy=multi-user.target
SVCEOF

      systemctl daemon-reload
      systemctl enable celery-training
      systemctl start celery-training
    EOF
  }

  service_account {
    scopes = ["cloud-platform"]
  }
}

output "training_engine_ip" {
  value = google_compute_instance.training_engine.network_interface[0].access_config[0].nat_ip
}

