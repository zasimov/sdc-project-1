all: collect loaders

collect::
	python3.5 pipeline.py

writeup.pdf: writeup.md
	pandoc --from=markdown --to=latex writeup.md -o writeup.pdf

loaders: gen_load_stats.py stats.csv rgb_stats.csv wrong_canny_stats.csv
	python gen_load_stats.py > load_stats.m
	python gen_load_stats.py rgb_stats.csv rgb_ > load_rgb_stats.m
	python gen_load_stats.py wrong_canny_stats.csv wrong_canny_ > load_wrong_canny_stats.m
	python gen_load_stats.py wrong_hough_stats.csv wrong_hough_ > load_wrong_hough_stats.m
