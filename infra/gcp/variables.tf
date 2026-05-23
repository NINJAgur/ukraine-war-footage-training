variable "project_id" { type = string }
variable "region"     { default = "us-central1" }
variable "zone"       { default = "us-central1-a" }
variable "bucket_name"{ default = "ukraine-footage-media" }

variable "postgres_password" {
  type      = string
  sensitive = true
}

variable "cors_origins" {
  type    = string
  default = "*"
}
