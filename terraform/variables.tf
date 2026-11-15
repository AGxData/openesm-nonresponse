variable "aws_region" {
  description = "AWS region for this project"
  type        = string
  default     = "us-east-2"
}

variable "aws_profile" {
  description = "AWS CLI profile used by Terraform"
  type        = string
  default     = "openesm-terraform"
}

variable "project_name" {
  description = "Project identifier"
  type        = string
  default     = "openesm-nonresponse"
}

variable "environment" {
  description = "Deployment environment"
  type        = string
  default     = "dev"
}