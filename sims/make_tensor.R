library(tidyverse)
library(nanoparquet)
library(fastDummies)
cells_df <- read_parquet("sims/dl_data/300_cells_df.parquet.gzip")

colnames(cells_df)[apply(cells_df,2,var) == 0]

min_dim <- -400
max_dim <- 400
nx <- ny <- 40
timestep <- 50

unique_types <- c(
  'lung_cell',
  'cancer_cell',
  'CD8_Tcell',
  'macrophage',
  'DC',
  'CD4_Tcell'
)

phases <- c(
  'G0G1_phase',
  'G2_phase',
  'S_phase',
  'M_phase',
  'apoptotic'
)

cells_df <- cells_df %>%
  mutate(cell_type = factor(cell_type,levels=unique_types),
         current_phase = factor(current_phase,levels=phases))

cells_df <- dummy_cols(cells_df, remove_selected_columns = TRUE)

cells_df

x <- cells_df$position_x
y <- cells_df$position_y
image_ids <- cells_df$timestep

# Creating the pixel boundaries
x_bins <- seq(min_dim, max_dim, length.out = nx + 1)
y_bins <- seq(min_dim, max_dim, length.out = ny + 1)

# Assigning each point to a pixel
x_pixel <- cut(x, breaks = x_bins, labels = FALSE, include.lowest = TRUE)
y_pixel <- cut(y, breaks = y_bins, labels = FALSE, include.lowest = TRUE)

# Create a dataframe with the pixel and image information
data <- cells_df %>%
  mutate(x_pixel=x_pixel,
         y_pixel=y_pixel) %>%
  rename(image_id=timestep) %>%
  select(-c(position_x,position_y,sensitivity_to_TNF_chemotaxis,sensitivity_to_debris_chemotaxis,debris_secretion_rate,activated_TNF_secretion_rate,ID))

# Counting the number of points of each type in each pixel for each image
data %>%
  pivot_longer(-c(x_pixel,y_pixel,image_id)) %>%
  dplyr::group_by(image_id, x_pixel, y_pixel, name) %>%
  dplyr::summarise(count = sum(value), .groups = 'drop') -> pixel_counts

# Filling missing combinations with NA
full_grid <- expand_grid(
  image_id = unique(image_ids),
  name = unique(pixel_counts$name),
  x_pixel = 1:nx,
  y_pixel = 1:ny
)

# Merge with the counts, ensuring missing combinations are NA
final_df <- full_grid %>%
  dplyr::left_join(pixel_counts, by = c("image_id", "x_pixel", "y_pixel","name")) %>%
  dplyr::arrange(image_id, x_pixel, y_pixel,name) %>%
  mutate(count=ifelse(is.na(count),0,count))

data_layers <- unique(final_df$name)
timesteps <- unique(final_df$image_id)

arrs <- lapply(timesteps,\(step) {
  mats <- lapply(data_layers,\(layer) {
    mat <- final_df %>%
      filter(image_id == step) %>%
      select(-image_id) %>%
      filter(name == layer) %>% 
      select(-name) %>%
      pivot_wider(names_from = x_pixel,values_from = count) %>%
      replace(is.na(.),0) %>%
      select(-y_pixel) %>%
      as.matrix()
  })
  
  arr <- simplify2array(mats)
})

arr <- simplify2array(arrs)
dim(arr)

data_layers
image(arr[,,1,50])


conc_df <- read_parquet("sims/dl_data/300_conc_df.parquet.gzip")
conc_df <-  conc_df %>%
  pivot_longer(-c(mesh_center_m,mesh_center_n,timestep))

conc_layers <- unique(conc_df$name)

conc_arrs <- lapply(timesteps,\(step) {
  mats <- lapply(conc_layers,\(layer) {
    mat <- conc_df %>%
      filter(timestep == step) %>%
      select(-timestep) %>%
      filter(name == layer) %>% 
      select(-name) %>%
      pivot_wider(names_from = mesh_center_m,values_from = value) %>%
      replace(is.na(.),0) %>%
      select(-y_pixel) %>%
      as.matrix()
  })
  
  arr <- simplify2array(mats)
})



