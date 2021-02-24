


resource "aws_lambda_function" "this" {
  function_name = "mtg-search-${local.slug}-${terraform.workspace}"
  image_uri     = "${data.aws_caller_identity.current.account_id}.dkr.ecr.us-east-1.amazonaws.com/mtg-search:${local.version}-${terraform.workspace}"
  package_type  = "Image"
  role          = aws_iam_role.lambda.arn

  timeout     = 60
  memory_size = 1024

}

resource "aws_iam_role" "lambda" {

  name = "mtg-search-${terraform.workspace}-r"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Sid    = ""
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })

}

resource "aws_lambda_permission" "this" {
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.this.function_name
  principal     = "apigateway.amazonaws.com"
}

