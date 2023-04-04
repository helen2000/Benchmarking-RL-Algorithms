:: boring constants
set IMAGE_NAME=rlcw
set CONTAINER_NAME=rlcw
set CONTAINER_BASE_PATH=/app

:: actually run the program
docker build --tag=%IMAGE_NAME% .
docker run --gpus all --memory="8g" --name=%CONTAINER_NAME% %IMAGE_NAME%

:: copy everything else over
:: help from here: https://stackoverflow.com/questions/3827567/how-to-get-the-path-of-the-batch-script-in-windows

docker cp %CONTAINER_NAME%:%CONTAINER_BASE_PATH%/out %~dp0

:: cleanup
docker container stop rlcw
docker container rm rlcw --force
docker image prune --force

:: stops console from automatically exiting upon completion
cmd -k

