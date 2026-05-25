terraform {
  required_providers {
    google = { source = "hashicorp/google", version = "~> 5.0" }
    null   = { source = "hashicorp/null",   version = "~> 3.0" }
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

resource "null_resource" "upload_weights" {
  triggers = {
    bucket = google_storage_bucket.media.name
  }

  provisioner "local-exec" {
    interpreter = ["bash", "-c"]
    command     = "PYTHONHTTPSVERIFY=0 '${path.module}/../../venv/Scripts/python.exe' '${path.module}/upload_weights.py' '${google_storage_bucket.media.name}' '${path.module}/../..'"
  }

  depends_on = [google_storage_bucket_iam_member.public_read]
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
    startup-script = <<-EOF
      #!/bin/bash
      set -e
      exec > /var/log/startup-script.log 2>&1

      # Swap — prevents OOM during Docker builds on e2-micro
      fallocate -l 2G /swapfile
      chmod 600 /swapfile
      mkswap /swapfile
      swapon /swapfile
      echo '/swapfile none swap sw 0 0' >> /etc/fstab

      # Docker + git
      apt-get update -qq
      apt-get install -y -qq docker.io docker-compose-v2 git
      systemctl enable docker
      systemctl start docker

      # Clone repo
      git clone --depth=1 https://github.com/NINJAgur/ukraine-war-footage-training.git /home/ubuntu/app
      chown -R ubuntu:ubuntu /home/ubuntu/app

      # Write .env
      cat > /home/ubuntu/app/.env <<ENVEOF
      POSTGRES_PASSWORD=${var.postgres_password}
      CORS_ORIGINS=${var.cors_origins}
      STORAGE_MODE=remote
      REMOTE_STORAGE_BUCKET=ukraine-footage-media
      ENVEOF

      # Build and start all CPU services
      cd /home/ubuntu/app
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

# ── T4 Spot VM (GPU worker) ───────────────────────────────────────────

resource "google_compute_disk" "datasets" {
  name = "ukraine-footage-datasets"
  zone = var.zone
  type = "pd-standard"
  size = 150
}

resource "google_compute_resource_policy" "gpu_schedule" {
  name   = "gpu-daily-schedule"
  region = var.region

  instance_schedule_policy {
    vm_start_schedule { schedule = "0 2 * * *" }
    vm_stop_schedule  { schedule = "0 5 * * *" }
    time_zone         = "UTC"
  }
}

resource "google_compute_instance" "t4_gpu" {
  name                      = "ukraine-footage-gpu"
  machine_type              = "n1-standard-1"
  zone                      = var.zone
  tags                      = ["gpu-worker"]
  allow_stopping_for_update = true

  scheduling {
    preemptible         = true
    automatic_restart   = false
    on_host_maintenance = "TERMINATE"
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

  attached_disk {
    source      = google_compute_disk.datasets.self_link
    device_name = "datasets"
  }

  network_interface {
    network = "default"
    access_config {}
  }

  resource_policies = [google_compute_resource_policy.gpu_schedule.id]

  metadata = {
    startup-script = <<-EOF
      #!/bin/bash
      set -e
      exec >> /var/log/startup-script.log 2>&1

      # Install NVIDIA driver on first boot, then reboot to load kernel module
      if [ ! -f /var/lib/nvidia-driver-installed ]; then
        apt-get update -qq
        # gcc-12 required: GCP kernel 6.8 built with gcc-12, DKMS must match
        apt-get install -y --no-install-recommends \
          build-essential gcc-12 linux-headers-$(uname -r) \
          ubuntu-drivers-common git python3 python3-venv python3-pip wget ffmpeg
        update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 12
        ubuntu-drivers autoinstall
        touch /var/lib/nvidia-driver-installed
        reboot
        exit 0
      fi

      # Wait for NVIDIA driver (available after reboot)
      until nvidia-smi &>/dev/null; do sleep 10; done

      # Sparse clone on first boot only
      if [ ! -d /home/ubuntu/app ]; then
        git clone --depth=1 --filter=blob:none --sparse \
          https://github.com/NINJAgur/ukraine-war-footage-training.git \
          /home/ubuntu/app
        cd /home/ubuntu/app
        git sparse-checkout set ml-engine shared
        chown -R ubuntu:ubuntu /home/ubuntu/app
      fi

      # Python venv + deps
      cd /home/ubuntu/app/ml-engine
      if [ ! -f venv/bin/celery ]; then
        python3 -m venv venv
        venv/bin/pip install -r requirements.txt
      fi

      # Download weights from GCS (re-runs if any weight is missing)
      if [ ! -f /home/ubuntu/app/ml-engine/runs/baseline/GENERAL/baseline_GENERAL_30/weights/best.pt ]; then
        pip3 install --quiet google-cloud-storage
        python3 - <<PYEOF
from google.cloud import storage
import pathlib
client = storage.Client()
ml_root = pathlib.Path("/home/ubuntu/app/ml-engine")
for blob in client.list_blobs("ukraine-footage-media", prefix="runs/"):
    if not blob.name.endswith("best.pt"):
        continue
    local = ml_root / blob.name
    local.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(local))
    print(f"Downloaded {blob.name}")
PYEOF
      fi

      # Mount persistent datasets disk
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

      # Swap on persistent disk — prevents OOM during large dataset extraction
      if [ ! -f "$DATASETS_MNT/swapfile" ]; then
        fallocate -l 4G "$DATASETS_MNT/swapfile"
        chmod 600 "$DATASETS_MNT/swapfile"
        mkswap "$DATASETS_MNT/swapfile"
      fi
      swapon "$DATASETS_MNT/swapfile" 2>/dev/null || true

      # Redirect kagglehub cache to persistent disk via symlink (env var unreliable across sudo)
      mkdir -p "$DATASETS_MNT/.cache/kagglehub"
      mkdir -p /home/ubuntu/.cache
      chown -R ubuntu:ubuntu "$DATASETS_MNT/.cache" /home/ubuntu/.cache
      ln -sfn "$DATASETS_MNT/.cache/kagglehub" /home/ubuntu/.cache/kagglehub

      # Symlink ml-engine/media → persistent disk so all paths stay identical
      mkdir -p "$DATASETS_MNT/media"
      chown -R ubuntu:ubuntu "$DATASETS_MNT"
      if [ ! -L /home/ubuntu/app/ml-engine/media ]; then
        rm -rf /home/ubuntu/app/ml-engine/media
        ln -s "$DATASETS_MNT/media" /home/ubuntu/app/ml-engine/media
      fi

      # Download Kaggle datasets + build merged folders (once — persists on disk)
      # Non-fatal: Celery still starts for annotation even if datasets aren't ready
      cd /home/ubuntu/app/ml-engine
      sudo -u ubuntu bash scripts/setup_datasets.sh || \
        echo "[startup] Dataset setup incomplete — will retry on next boot"

      # Write .env (refreshed every boot in case IPs change)
      cat > /home/ubuntu/app/.env <<ENVEOF
DATABASE_SYNC_URL=postgresql+psycopg2://postgres:${var.postgres_password}@${google_compute_instance.e2_micro.network_interface[0].network_ip}:5432/ukraine_footage
CELERY_BROKER_URL=redis://${google_compute_instance.e2_micro.network_interface[0].network_ip}:6379/0
CELERY_RESULT_BACKEND=redis://${google_compute_instance.e2_micro.network_interface[0].network_ip}:6379/1
REDIS_URL=redis://${google_compute_instance.e2_micro.network_interface[0].network_ip}:6379/0
STORAGE_MODE=remote
REMOTE_STORAGE_BUCKET=ukraine-footage-media
ENVEOF
      chown ubuntu:ubuntu /home/ubuntu/app/.env

      # systemd service
      cat > /etc/systemd/system/celery-gpu.service <<SVCEOF
[Unit]
Description=Celery GPU Worker
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/app/ml-engine
EnvironmentFile=/home/ubuntu/app/.env
ExecStart=/home/ubuntu/app/ml-engine/venv/bin/celery -A celery_app worker -Q gpu --concurrency=1 --loglevel=info --beat
Restart=on-failure
RestartSec=30
StandardOutput=append:/var/log/celery-gpu.log
StandardError=append:/var/log/celery-gpu.log

[Install]
WantedBy=multi-user.target
SVCEOF

      systemctl daemon-reload
      systemctl enable celery-gpu
      systemctl start celery-gpu
    EOF
  }

  service_account {
    scopes = ["cloud-platform"]
  }
}

output "t4_gpu_ip" {
  value = google_compute_instance.t4_gpu.network_interface[0].access_config[0].nat_ip
}
