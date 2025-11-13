pkgroot <- Sys.getenv("OMICSTL_PKG_ROOT")

suppressMessages({
  suppressPackageStartupMessages({
    suppressWarnings({
      if (!require(here)) {
        install.packages("here", quiet = TRUE)
      }
    })
  })
})

if (pkgroot == "") {
  pkgroot <- here::here()
}

required_packages <- c(
	"viRandomForests",
  #"ggplot2","Matrix", "data.table", "survival", "Rcpp", "readr"
	"randomForest",
	"tibble",
	"moments",
	"dplyr",
	"furrr",
	"progressr"
)
options(repos = c(CRAN = "https://cloud.r-project.org/"))

options(warn = -1)
suppressMessages({
  suppressPackageStartupMessages({
    suppressWarnings({
      for (pkg in required_packages) {
        if (!require(pkg, character.only = TRUE)) {
          if (pkg == "viRandomForests") {
            path <- file.path(pkgroot, "src", "omicstl", "r", "viRF_code", "viRandomForests_1.0.tar.gz")
            install.packages(path,
                            repos = NULL,
                            type = "source",
                            quiet = TRUE)
          } else if (pkg == "mvdalab") {
            BiocManager::install(pkg, quiet = TRUE)
          } else {
            install.packages(pkg, dependencies = TRUE, quiet = TRUE)
          }
        }
      }
    })
  })
})

invisible(
  suppressPackageStartupMessages(
    lapply(required_packages, library, character.only = TRUE)
  )
)

options(warn = 0)
