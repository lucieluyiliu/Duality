library(shiny)
library(plotly)

## ----- Moments (swap in your own) -----
mu <- c(1.06, 1.04, 1.08, 1.03)  # gross means
n  <- length(mu)
set.seed(42)
A <- matrix(rnorm(n*n), n, n)
Sigma <- A %*% t(A)
Sigma <- Sigma / (max(diag(Sigma)) * 150)
one <- rep(1, n)
SigInv <- solve(Sigma)

## ----- Helpers -----
mv_frontier_sd <- function(R){
  A <- as.numeric(t(one) %*% SigInv %*% one)
  B <- as.numeric(t(one) %*% SigInv %*% mu)
  C <- as.numeric(t(mu)  %*% SigInv %*% mu)
  D <- A*C - B^2
  sqrt((A*R^2 - 2*B*R + C)/D)
}
tangency_point <- function(Rf){
  z <- SigInv %*% (mu - Rf*one)
  if(all(abs(z) < 1e-12)){
    w <- (SigInv %*% one) / as.numeric(t(one) %*% SigInv %*% one)   # GMV fallback
  } else {
    w <- z / as.numeric(t(one) %*% z)
  }
  mean <- as.numeric(t(w) %*% mu)
  sd   <- sqrt(as.numeric(t(w) %*% Sigma %*% w))
  c(sd = sd, mean = mean)
}
hj_sigma_min <- function(mbar){
  v <- one - mbar*mu
  sqrt(as.numeric(t(v) %*% SigInv %*% v))
}

## ----- Static grids -----
# GMV
gmv_w  <- (SigInv %*% one) / as.numeric(t(one) %*% SigInv %*% one)
gmv_er <- as.numeric(t(gmv_w) %*% mu)

# Build a dense R-grid, compute (sd, er)
R_grid <- seq(0.95*min(min(mu), gmv_er),
              1.05*max(max(mu), gmv_er), length.out = 1200)
MV_all <- data.frame(sd = mv_frontier_sd(R_grid), er = R_grid)

# Split by ER relative to GMV, order each half by sd increasing
MV_left  <- subset(MV_all, er <= gmv_er)
MV_right <- subset(MV_all, er >= gmv_er)
MV_left  <- MV_left[order(MV_left$sd), ]
MV_right <- MV_right[order(MV_right$sd), ]

# Stitch into ONE continuous path: left→GMV→right (drop duplicate GMV row)
MV_curve <- rbind(MV_left, MV_right[-1, , drop = FALSE])

# HJ frontier grid
HJ_axis <- seq(0.6, 1.4, length.out = 400)
HJ <- data.frame(mbar = HJ_axis, sd = sapply(HJ_axis, hj_sigma_min))

## ----- UI -----
ui <- fluidPage(
  titlePanel("MV–HJ Duality"),
  sidebarLayout(
    sidebarPanel(
      sliderInput("mbar", "E[m] (moves both panels; Rf = 1/E[m])",
                  min = 0.70, max = 1.30, value = 1.00, step = 0.005, width = "100%")
    ),
    mainPanel(
      plotlyOutput("mvPlot", height = "380px"),
      plotlyOutput("hjPlot", height = "360px")
    )
  )
)

## ----- Server -----
server <- function(input, output, session){
  
  output$mvPlot <- renderPlotly({
    Rf <- 1.0 / input$mbar
    tp <- tangency_point(Rf)
    
    plot_ly() |>
      add_lines(data = MV_curve, x = ~sd, y = ~er, name = "MV Frontier") |>
      add_markers(x = tp["sd"], y = tp["mean"], name = "Tangency", marker = list(size = 9)) |>
      add_markers(x = mv_frontier_sd(gmv_er), y = gmv_er, name = "GMV",
                  marker = list(size = 8, symbol = "diamond")) |>
      layout(
        title = paste0("Top: Rf≈", round(Rf,3),
                       "  →  Tangency (sd≈", round(tp["sd"],3),
                       ", E[R]≈", round(tp["mean"],3), ")"),
        xaxis = list(title = "Portfolio SD"),
        yaxis = list(title = "Expected Gross Return")
      )
  })
  
  output$hjPlot <- renderPlotly({
    mbar <- input$mbar
    sdm  <- hj_sigma_min(mbar)
    
    plot_ly() |>
      add_lines(data = HJ, x = ~mbar, y = ~sd, name = "HJ Frontier") |>
      add_markers(x = mbar, y = sdm, name = "Current E[m]", marker = list(size = 9)) |>
      layout(
        title = paste0("Bottom: min σ(m) at E[m] = ", round(mbar,3),
                       "  ≈  ", round(sdm,3)),
        xaxis = list(title = "E[m]"),
        yaxis = list(title = "σ(m)")
      )
  })
}

shinyApp(ui, server)
