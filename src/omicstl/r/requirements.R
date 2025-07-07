required_packages <- c(
	"randomForest",
	"tibble",
	"viRandomForests",
	"moments",
	"dplyr",
	"furrr",
	"progressr"
	)
options(repos = c(CRAN = "https://cloud.r-project.org/"))

options(warn = -1)
suppressPackageStartupMessages({
  suppressWarnings({
    for (pkg in required_packages) {
      if (!require(pkg, character.only = TRUE)) {
        if (pkg == "viRandomForests") {
          path <- here::here("r", "viRF_code", "viRandomForests_1.0.tar.gz")
          install.packages(path,
                           repos = NULL,
                           type = "source",
                           quiet = TRUE)
          next
        }
        install.packages(pkg, dependencies = TRUE, quiet = TRUE)
      }
    }
  })
})

invisible(
  suppressPackageStartupMessages(
    lapply(required_packages, library, character.only = TRUE)
  )
)

options(warn = 0)
