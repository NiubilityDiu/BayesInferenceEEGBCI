library(ggplot2)
library(gridExtra)

target_letters = c('T', 'H', 'E', '_',
                   'Q', 'U', 'I', 'C', 'K', '_',
                   'B', 'R', 'O', 'W', 'N', '_',
                   'F', 'O', 'X')

target_letter_nums = c(20, 8, 5, 36, 
                       17, 21, 9, 3, 11, 36,
                      2, 18, 15, 23, 14, 36,
                      6, 15, 24)
target_letter_rows = c(4, 2, 1, 6, 3,
                       4 ,2 ,1, 2, 6,
                       1, 3, 3, 4, 3, 
                       6, 1, 3, 4)
target_letter_cols = c(8, 8, 11, 12, 11,
                       9, 9, 9, 11, 12, 
                       8, 12, 9, 11, 8,
                       12, 12, 9, 12)
num_letter = length(target_letters)
K_num_ids = c(106:108, 111:115, 117:123, 143, 145:147,
              151, 152, 154, 155, 156, 158, 159, 160,
              166, 167, 171, 172, 177, 178, 179,
              183, 184, 185, 190, 191, 212, 223)

# candidate_letters = c(toupper(letters), 1:5, 'SPEAK', '.', 'BS', '!', '_')

# n_length_fit = 20
# dec_factor = 4
# time_length = n_length_fit / 32 * 1000  # down-sampling latency time length

num_electrode = 16

channel_name_short = c('F3', 'Fz', 'F4', 'T7', 'C3', 'Cz', 'C4', 'T8',
                       'CP3', 'CP4', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz')
