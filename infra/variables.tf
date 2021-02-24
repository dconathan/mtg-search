
locals {
  # load and extract version from mtg_search.version
  version = regex("\\d+.\\d+.\\d+", file("${path.module}/../mtg_search/version.py")) # e.g. 0.1.2
  slug    = replace(local.version, ".", "-")                                         # e.g. 0-1-2
}

output "version" {
  value = local.version
}

output "slug" {
  value = local.slug
}