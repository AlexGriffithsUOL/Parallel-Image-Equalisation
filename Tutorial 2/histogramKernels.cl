kernkernel void histogramEqualisation(global const uchar* A, global uchar* B, global int* correspondingArr) {
	int id = get_global_id(0);
	B[id] = correspondingArr[A[id]];

}

kernel void vector_add(global const unsigned int* A, global const unsigned int* B, global int* C) {
	int width = get_global_size(0); //image width in pixels
	int height = get_global_size(1); //image height in pixels
	int image_size = width * height; //image size in pixels
	int channels = get_global_size(2); //number of colour channels: 3 for RGB

	int x = get_global_id(0); //current x coord.
	int y = get_global_id(1); //current y coord.
	int c = get_global_id(2); //current colour channel

	int id = x + y * width; +c * image_size; //global id in 1D space

	int key = (int)A[id];
	++C[key];
}