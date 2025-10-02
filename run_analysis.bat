@echo off
echo [Step 1/2] Running MCR-ALS analysis with plot generation...
python run_main.py ^
    --file_path "data/TAS/TA_Average-crop-TCA-tzs-tzs-tzs-chirp.csv" ^
    --n_components 3 ^
    --wavelength_range 420 750 ^
    --delay_range 0.01 10 ^
    --save_plots ^
    --save_results ^
    --output_dir results

if %errorlevel% neq 0 (
    echo MCR-ALS analysis failed.
    exit /b %errorlevel%
)

echo.
echo [Step 2/2] Running Global Fit analysis workflow...
python Globalfit/examples/auto_workflow.py --mcr_results results

if %errorlevel% neq 0 (
    echo Global Fit analysis failed.
    exit /b %errorlevel%
)

echo.
echo Analysis complete. All results are saved in the 'results' directory.
