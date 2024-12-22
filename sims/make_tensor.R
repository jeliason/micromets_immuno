library(tidyverse)
library(nanoparquet)
library(fastDummies)
library(reticulate)

min_dim <- -400
max_dim <- 400
nx <- ny <- 40

SYSTEM_ENV <- Sys.getenv("SYSTEM_ENV")

if(SYSTEM_ENV == "laptop") {
  use_condaenv("physicell")
} else {
  use_virtualenv("~/virtual_envs/physicell")
}

utils <- reticulate::import_from_path("utils","sims/src/")

OUTPUT_PATH <- utils$get_output_path()

make_array <- \(sim_id) {
  print(sim_id)
  cells_df_file <- paste0(OUTPUT_PATH,sim_id,"_cells_df.parquet.gzip")
  conc_df_file <- paste0(OUTPUT_PATH,sim_id,"_conc_df.parquet.gzip")
  cells_df <- read_parquet(cells_df_file)

  
  unique_types <- utils$get_cell_types()
  phases <- utils$get_phases()
  
  cells_df <- cells_df %>%
    mutate(cell_type = factor(cell_type,levels=unique_types),
           current_phase = factor(current_phase,levels=phases))
  
  cells_df <- dummy_cols(cells_df, remove_selected_columns = TRUE)
  
  cells_df
  
  x <- cells_df$position_x
  y <- cells_df$position_y
  timestep <- cells_df$timestep
  
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
    select(-c(position_x,position_y))
  
  # Counting the number of points of each type in each pixel for each image
  data %>%
    pivot_longer(-c(x_pixel,y_pixel,timestep)) %>%
    dplyr::group_by(timestep, x_pixel, y_pixel, name) %>%
    dplyr::summarise(count = sum(value), .groups = 'drop') -> pixel_counts
  
  # Filling missing combinations with NA
  full_grid <- expand_grid(
    timestep = unique(timestep),
    name = unique(pixel_counts$name),
    x_pixel = 1:nx,
    y_pixel = 1:ny
  )
  
  # Merge with the counts, ensuring missing combinations are NA
  final_df <- full_grid %>%
    dplyr::left_join(pixel_counts, by = c("timestep", "x_pixel", "y_pixel","name")) %>%
    dplyr::arrange(timestep, x_pixel, y_pixel,name) %>%
    mutate(count=ifelse(is.na(count),0,count))
  
  data_layers <- unique(final_df$name)
  timesteps <- unique(final_df$timestep)
  
  arrs <- lapply(timesteps,\(step) {
    mats <- lapply(data_layers,\(layer) {
      mat <- final_df %>%
        filter(timestep == step) %>%
        select(-timestep) %>%
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
  # dim(arr)
  # 
  # data_layers
  # image(arr[,,1,50])
  
  
  conc_df <- read_parquet(conc_df_file)
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
        select(-mesh_center_n) %>%
        as.matrix()
    })
    
    arr <- simplify2array(mats)
  })
  
  conc_arr <- simplify2array(conc_arrs)
  
  final_arr <- abind::abind(arr,conc_arr,along=3)
}


# np <- reticulate::import("numpy")
# gzip <- reticulate::import("gzip")
# tensor_path <- paste0(OUTPUT_PATH,"data.npy.gz")
# np$save(tensor_path,final_arr)
# 
# f <- gzip$GzipFile(tensor_path, "w")
# np$save(file=f, arr=final_arr)
# f$close()
# 
# 
# f <- gzip$GzipFile(tensor_path, "r")
# final_arr2 <- np$load(f)
# f$close()
# 
# dim(final_arr2)
# 
# final_arr_py <- r_to_py(final_arr)
# np$savez_compressed(paste0(OUTPUT_PATH,"data_z.npz"),final_arr_py

args <- commandArgs(trailingOnly = TRUE)
sim_id <- args[1]
final_arr <- make_array(sim_id)

h5py <- import("h5py")

# Save to HDF5 file
# Save to HDF5 file
f <- h5py$File(paste0(OUTPUT_PATH,sim_id,"_data.h5"), "w")  # Open file in write mode
f$create_dataset(paste0("final_arr_",sim_id), data = final_arr, compression = "gzip", compression_opts = 9)  # Save dataset
f$close() 
