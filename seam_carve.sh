# ./seam_carve.sh img_in img_out X Y

g++-15 -O3 -lm --openmp transpose.cpp -o b.out
g++-15 -O3 -lm --openmp carve.cpp -o a.out

./a.out $1 carvedX.jpg $3
./b.out carvedX.jpg carvedX.jpg
./a.out carvedX.jpg $2 $4
./b.out $2 $2

rm carvedX.jpg