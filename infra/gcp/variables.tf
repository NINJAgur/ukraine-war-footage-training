variable "project_id"   { type = string }
variable "region"           { default = "us-central1" }
variable "zone"             { default = "us-central1-a" }
variable "inference_zone"   { default = "us-central1-a" }
variable "bucket_name"  { default = "ukraine-footage-media" }

variable "postgres_password" {
  type      = string
  sensitive = true
}

variable "jwt_secret" {
  type      = string
  sensitive = true
}

variable "admin_username" {
  type    = string
  default = "admin"
}

variable "admin_password" {
  type      = string
  sensitive = true
}

variable "cors_origins" {
  type    = string
  default = "*"
}

variable "ssh_deploy_key_pub" {
  type        = string
  default     = ""
  description = "ed25519 public key added to ubuntu authorized_keys on e2-micro and inference-engine for CI/CD SSH deploys. Generate: ssh-keygen -t ed25519 -C 'github-actions' -f ~/.ssh/gcp-deploy"
}
