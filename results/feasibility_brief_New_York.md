# Feasibility notes for 6 November

These checks describe the processed New York sample already on disk. They do not re-download the UCSD Google Local files, rerun sampling, or fit embeddings or clustering models.

## Data frame

The business file contains 10,000 sampled businesses, all with coordinates. The review file contains 139,727 retained reviews. Every retained review matches a sampled business. Preprocessing, as recorded in the existing coverage file, drew a uniform sample of 10,000 businesses from the NYC bounding box with seed 42, then kept at most 20 reviews per business with a reservoir. Eligible reviews for that reservoir were reviews with text or an image URL.

9,826 businesses have at least one retained review. 174 have none. Those 174 still count in business-level denominators and in the coordinate plot. They add nothing to review or image-URL counts.

The sampling window is latitude 40.48–40.93 and longitude −74.26 to −73.68. It is the box already used in preprocessing. It is not the city boundary. Sampled coordinates span latitude 40.507–40.930 and longitude -74.253–-73.680.

## Geographic coverage

No census-tract, borough, or neighborhood boundary file is in the repository. The units below are exploratory grids aligned to the sampling box. A grid cell is not a neighborhood.

The finer grid has 25 rows and 25 columns. Each cell is 0.018° of latitude by 0.0232° of longitude, about 2.0 km by 2.0 km at 40.7°N. The coarser check has 9 rows and 10 columns, 0.050° by 0.058°, about 5.6 km by 4.9 km. Degree steps differ because a degree of longitude is shorter than a degree of latitude at this latitude. Cell land area is still unequal: the box contains water and some land outside New York City.

On the 25×25 grid, 287 of 625 cells contain a sampled business and 338 (54.1%) are empty. Among occupied cells the median count is 16 and the maximum is 546, in the cell centered near latitude 40.723, longitude -73.993. The densest tenth of occupied cells contains 48.2% of the businesses that fall in occupied cells. The Gini coefficient of business counts is 0.83 across all cells and 0.63 across occupied cells.

The 9×10 grid is less sparse and still uneven: 33 of 90 cells are empty (36.7%), the occupied median is 71, the maximum is 1,627, and the all-cell Gini coefficient is 0.78. Empty cells are an expected mix of water and land with no sampled business. They are not measurements of neighborhoods with zero activity.

Businesses were sampled uniformly from listed businesses, not uniformly across land. Commercial corridors with many listings contribute more points than residential blocks. That is spatial sampling bias relative to a map of places. Half-box shares, which only describe the point cloud, put 33.3% of businesses in the southern half of the sampling box and 40.8% in the western half.

Retained review counts are also uneven, and they are capped. 5,168 businesses are at the 20-review cap. The median `num_of_reviews` stored on the business record is 36, and 60.7% of businesses with a listed count are above 20. The retained file is a capped subset of eligible reviews, not the full Google review history. Image-bearing review counts use that same cap.

## Image coverage and bias

Image URL occurrences, unique URLs, reviews with images, and businesses with images are separate counts:

- Image URL occurrences in retained reviews: 23,757
- Unique image URLs: 23,595
- Retained reviews with at least one image URL: 8,240 (5.9% of retained reviews)
- Businesses with at least one image-bearing retained review: 4,139 (41.4% of the 10,000 sampled businesses)

Occurrences and unique URLs differ by 162, so repeated use of the same URL is uncommon in the retained sample. A business can clear the business-level threshold with a single image-bearing review. That is why 41.4% of businesses have an image while 5.9% of retained reviews do.

The business-level percentage uses every sampled business as the denominator, including businesses with no retained review. Among businesses that have at least one retained review, the share is 42.1%. The review-level percentage uses retained reviews as the denominator and answers a different question.

Per sampled business, the number of image-bearing retained reviews has median 0, mean 0.82, and 90th percentile 2. 58.6% have zero. Among the businesses with at least one, the median is 1 and the maximum is 13. The cap of 20 retained reviews bounds that maximum.

Counts of image-bearing reviews follow the same concentrated pattern as the business counts. The share of businesses with an image does not. On the 25×25 grid, 127 cells have at least 20 businesses. Their business-level image share ranges from 15.4% to 60.8% (median 38.1%). On the 9×10 grid, 40 cells have at least 30 businesses, and the share ranges from 21.6% to 51.7% (median 37.6%). Gray cells on the rate maps are below those cutoffs. The sample-wide business rate is 41.4%, so the geographic range is real but much smaller than the range in raw counts.

Categories are multi-label. 4,520 businesses have two or more labels, 31 have none, and there are 21,701 label assignments across 1,912 distinct labels (mean 2.17 labels per business). A business enters the denominator of every category it carries. Rows overlap and cannot be added into a partition of the 10,000 businesses. The category table keeps labels with at least 20 sampled businesses (235 labels; 1,677 rarer labels are omitted). The first-label columns use the first string in the stored list. That order is not a verified primary category.

Among labels with at least 100 sampled businesses (34 labels), the business-level image share ranges from 6.7% for Doctor (n=119) to 84.6% for Tourist attraction (n=123). The category figure is the 18 most common labels only, ordered by this percentage. Those labels run from 29.9% for Grocery store (n=194) to 71.3% for Cafe (n=202). First-label percentages are unstable when few businesses carry the label first. Breakfast restaurant is 51.8% across 170 businesses, but 80.0% among the 5 businesses for which it is the first stored label. The multi-label denominator is the one used here. These gaps are differences in image-URL presence. They are not evidence that image features improve a representation of neighborhoods.

## Limitations

The box and the grids do not follow official boundaries, and empty cells mix water with unsampled land. Uniform sampling of businesses over-represents places with many listings. Review retention is capped at 20 eligible reviews, so businesses with long review histories are only partly observed. Reviews with neither text nor an image URL were dropped before the reservoir, so the image rate among retained reviews is not the image rate among every stored review. This count treats a non-empty image URL as coverage. It does not open the file. A separate 200-URL download, already on disk, decoded 194 images and found 6 HTTP 404 responses, so URL presence is close to file availability in that small check and is not a census of pixels. No embedding or clustering model was trained. These results do not show that adding image features improves neighborhood representation.

## Files

- `results/figures/sampled_business_coordinates.png`
- `results/figures/exploratory_grid_25x25_counts.png`
- `results/figures/exploratory_grid_25x25_image_business_share.png`
- `results/figures/exploratory_grid_9x10_counts.png`
- `results/figures/exploratory_grid_9x10_image_business_share.png`
- `results/figures/image_bearing_reviews_per_business.png`
- `results/figures/image_coverage_by_category.png`
- `results/tables/exploratory_grid_25x25_New_York.csv`
- `results/tables/exploratory_grid_9x10_New_York.csv`
- `results/tables/category_image_coverage_New_York.csv`
- `results/tables/image_review_distribution_New_York.csv`
- `results/tables/feasibility_summary_New_York.json`
