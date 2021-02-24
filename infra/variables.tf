

locals {
        version = regex("\\d+.\\d+.\\d+", file("${path.module}/../mtg_search/version.py"))
        slug = replace(local.version, ".", "-")
}

output version {
        value = local.version
}

output slug {
        value = local.slug
}