
terraform {
  backend "s3" {
    bucket = "tfstate-347651660649-us-east-1"
    key    = "mtg-search.tfstate"
    region = "us-east-1"
  }
}

data "aws_caller_identity" "current" {}
