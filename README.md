# Crack-phase features

Python tool for analysis of cracks in heterogeneous surface. Repository contains image processing algorithms for analysis of electron microscopy images.

The algorithm works according to the schema:

<img width="663" alt="Screenshot 2025-02-18 at 16 24 29" src="https://github.com/user-attachments/assets/11608f33-58eb-452a-a27b-70e111e9051b" />

There are three steps:

- **Crack detection** - processes the SEM BSE images on which segment cracks.
- **Phase maps** - builds grains from SEM-EDS image
- **Grains / phases cracking analysis** - uses the crack mask and EDS images to fulfill grain cracking information.

To combine both pipeline 

## Crack detection
Applies two step thresholding for crack detection. 
First threshold creates seeds. 
The second threshold define areas where flood fill is applied.

This functionality covers `cracks/detection.py`. For usage see:
```bash
python -m cracks.detection --help
```

Generates:
- `crack_mask.png` (binary image) - {0, 1} for each pixel, stored as PNG
- user outputs with `[user]` prefix:
  - `[user]colored.png` - black are crack seed, red are crack flooded areas, gray and green are other pixels (gray are darker, green lighter)
  - `[user]crack-mask.png` - the same as crack mask but values are {0, 255}
  - `[user]input-grayscale.png` - input converted to grayscale image values in range \[0,255\]
  - `[user]input-normalized.png` - input with applied normalization. Normalization stretches values into {0, ..., 255} if they are not and center histogram around specified value. Each side of the histogram is stretched different way.

### Example

```bash
# Replace this variable with your prefered path
export INPUT_DATA_DIR = "${HOME}/crack-analysis/inputs"
export OUTPUT_FOLDER = "${HOME}/crack-analysis/outputs"

# If you want some example data call this section
mkdir -p ${INPUT_DATA_DIR}
hf download --repo-type=dataset research-centre-rez/crack-phase-sample --local-dir ${INPUT_DATA_DIR}

python -m cracks.detection -o ${OUTPUT_FOLDER} ${INPUT_DATA_DIR}/E14_10.tif
```

## Phase maps

Phase map is a special output from an electron microscope. Electron microscope produces a spectrum for each pixel. But spectra are very hard to visualize due to high number of dimensions. The spectra correspond to elements and elements can be encoded by colors. Product of this visualization technique we call **phase map** and it is fulfilled by phase definition (@see [config/phases_config.json]).

Phase maps are noisy for direct usage and contain segments with unclear phase. For this reason we do segmentation of phase map first with some morphology for fixing small segments. Segmentation is based on La*b* color distance of reference phase colors [@see phase config](config/phase_config.json) and color in the phase map (according to the closest phase distance the phase association is done).

This functionality is in `phase_map/__init__.py`. For usage see:
```bash
python -m phase_map --help
```

Generates:
- user outputs
  - `[user]grain_map_filtered.tiff` - 16bit int grayscale image where each grain has own id. Grains are filtered in this image, grains smaller than `grain_size_px_limit` pixels are dropped
  - `[user]grain_map.tiff` - as previous but before applying `grain_size_px_limit`
  - `[user]matrix_grains.png` - Binary image where 1 denotes matrix phase (after grain size filtration), 0 denotes non-matrix phase. Matrix denotes phase which is mixed from other phases.
  - `[user]non_matrix_grains.png` - Negative of the previous
  - `[user]phase_map-${PHASE["label"]}.png` - Binary image where 1 correspond with present of the `${PHASE}`
  - `[user]phase_map.png` - Visualization of the phase map after cleaning, i.e. segmented phase map.
- `grain_map_filtered.tiff` - 16bit int image where each grain has own ID after application of `grain_size_px_limit`
- `grain_map.tiff` - 16bit int image where each grain has own ID (no size filter is applied)
- `phase_img_lab.png` - phase map converted to La*b* color space (png should be interpreted as La*b* not RGB)
- `phase_map_adjusted.png` - if input phase map is lightness-adjusted this output is generated. This means that the lightness of the input is stretched to maximum.
- `phase_map_filtered.png` - grayscale image where each pixel is attached to one of the phases. New phase **matrix** is introduced with the highest ID. **Matrix** is a mixture of other phases with too small grains. However, the `grain_size_px_limit` is applied and all segments under limit are attached to **matrix** phase.
- `phase_map_gaussian_blur.png` - for handling nosiy phase edges the blur can be applied (@see parameter `--gaussian_blur`), this image is produces for checking how blurred image looks like
- `phase_map_rgb` - the input phase map image
- `phase_map` - same as `phase_map_filtered.png` but without applying `grain_size_px_limit`

### Example

Follow previous example with folder structure and hugging face download.

```bash
# Continuation of previous script

python -m phase_map ${INPUT_DATA_DIR}/E14_10.png ${OUTPUT_FOLDER}/crack-mask.png -o ${OUTPUT_FOLDER}
```

## Grains / phases cracking analysis 

Main part of the toolbox is analysis of cracks. This can be executed by `full_crack_analysis.py` script. For usage @see:
```bash
python full_crack_analysis.py --help
```
There are two possible flags implemented {crack_features, phase_features}. The first one (crack_features), is focused only on cracks itself (their length, max width, average width, distance between endpoints and length of their boundary). Second part (phase_features) puts cracks into context of phase map and measure wheather crack is between two phases or through some phase. This way we can quantify weak compounds of the scanned material (i.e. phases which crack more often or interface between two phases which is week.)

Generates:
- `cracks_features.csv` - statistics about the cracks containing:
  - label - i.e. crack ID which is used in `crack_mask.png`
  - crackSize_px - number of pixels belonging to the crack
  - skeleton_px - number of pixels belonging to the crack skeleton (medial axis)
  - maxWidth_px - the biggest distance from medial axis of the crack to the closest boundary point of the crack
  - avgWidth_px - average of the crack width (i.e. distance from medial axis to the boundary)
  - boundaryLength_px - crack boundary length
  - farthestPoints_px - distance of two points of the crack which have the biggest distance between them
- `phase-edge.png` - A stacked bar chart showing number of crack's skeleton pixels between each pair of phases
- `phase-through.png` - A bar chart showing number of crack's skeleton pixels going through a single phase
- `phases_cracks.csv` - raw data for the previous two charts (numbers of cracks skeletons pixels between each pair of phases as well as going through single phase).
- `skeleton-neighbors.csv` - full overview of the analysis. Each row of this table contains:
  - crack skeleton coord X - x position of evaluated skeleton pixel
  - crack skeleton coord Y - y position of evaluated skeleton pixel
  - crack ID - the ID of the crack used in `crack_mask.png`
  - branch ID - the ID of the crack branch (each crack is split into branches - a single pixel wide path)
  - gradient X - $\delta x$ in the skeleton point $[X, Y]$
  - gradient Y - $\delta y$ in the skeleton point $[X, Y]$
  - left neighbor X - x-coordinate of the first non-crack pixel in the direction perpendicular (clockwise) to gradient for the point $[X,Y]$
  - left neighbor Y - y-coordinate of the first non-crack pixel in the direction perpendicular (clockwise) to gradient for the point $[X,Y]$
  - right neighbor X - x-coordinate of the first non-crack pixel in the direction perpendicular (counter-clockwise) to gradient for the point $[X,Y]$
  - right neighbor Y - y-coordinate of the first non-crack pixel in the direction perpendicular (counter-clockwise) to gradient for the point $[X,Y]$
  - left neighbor phase ID - phase belonging to the left neighbor
  - right neighbor phase ID - phase belonging to the right neighbor

### Example

```bash
# Continuation of previous script
python full_crack_analysis.py -f crack_features --sample-name E14_10 -o ${OUTPUT_FOLDER} --phase-map-path ${OUTPUT_FOLDER}/phase_map_filtered.png ${OUTPUT_FOLDER}/crack-mask.png
python full_crack_analysis.py -f phase_features --sample-name E14_10 -o ${OUTPUT_FOLDER} --phase-map-path ${OUTPUT_FOLDER}/phase_map_filtered.png ${OUTPUT_FOLDER}/crack-mask.png
```


# Installation

### Local

Clone the repository, create a fresh virtual environment. Install the package dependencies and test dependencies with:

```bash
pip install -r requirements.txt pytest
```

## Running Tests

To run the full test suite with `pytest`:

```bash
pytest
```

To run tests with detailed verbosity:

```bash
pytest -v
```

The test suite covers:
- **Unit tests**: Morphological operations (`bw_morph`), skeleton derivates & branches, crack detection & feature extraction, phase map thresholding & grain segmentation, full crack analysis chart/csv generators, and utilities (`utils.config_loader`, `utils.image_logger`, `utils.output_dir_generator`).
- **Integration tests**: End-to-end execution of the CLI entry points (`python -m cracks.detection`, `python -m phase_map`, `python full_crack_analysis.py`) using synthetic test samples modeled on the microscopy data in `workspace/`.

### Docker

Build the image

```bash
docker build -t cracks .
```

Run it with

```bash
docker run --rm \
    -v ${INPUT_DATA_DIR}:/workspace \
    -v <path_to_data_on_host>:/data \
    -w /workspace \
    cracks \
    --root_dir /workspace<experiment_path>/
```

replacting `<experiment_path>` with the path to your prepared folder. The folder has the following structure.

## Experiment structure

- `after` - containting .npy stacks
- `before` - containting .npy stacks

## Output format

Contents created during each experiment run

`concrete_processed.csv` - cracks/bubbles measurements

- `group_name` - name of the subdirectory

- `skeletonSum_px` - skeleton area measurements for the group

- `crackBoundaryLengthSum_px` - boundary length measurements for the group

- `bubbleAreaSum_px` - bubble area measurements for the group

- `bubbleBoundaryLengthSum_px` - bubble boundary length measurements for the group

- `bubbleCount_px` - number of uniquely segmented bubbles in the group

For each processed file:

Synthetic IQR projection visualization
Segmented cracks (red) and bubbles (blue) masks

