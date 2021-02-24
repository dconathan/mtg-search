resource "aws_cloudwatch_log_group" "this" {
  name              = "/aws/lambda/mtg-search-${var.environment}"
  retention_in_days = 1
}

resource "aws_iam_policy" "logging" {
  name        = "lambda-logging-${var.environment}-p"
  path        = "/"
  description = "IAM policy for logging from a lambda"

  policy = jsonencode(
    { "Version" : "2012-10-17",
      "Statement" : [
        {
          Action : [
            "logs:CreateLogGroup",
            "logs:CreateLogStream",
            "logs:PutLogEvents"
          ],
          Resource : "arn:aws:logs:*:*:*",
          Effect : "Allow"
        }
      ]
    }
  )
}

resource "aws_iam_role_policy_attachment" "lambda_logs" {
  role       = aws_iam_role.lambda.name
  policy_arn = aws_iam_policy.logging.arn
}