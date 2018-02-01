#ifndef _EXPORT_
#define _EXPORT_

// export data to Javascript
void writeJS(mat3 * tab, vec2 * tabAmplitude, int N)
{
	ofstream file("results/ltc_tables_opt_2.js");

	file << "var g_ltc_mat_2 = [";

	int n = 0;
	for (int i = 0; i < N*N; ++i, n += 4)
	{
		const mat3& m = tab[i];

		float a = m[0][0];
		float b = m[0][2];
		float c = m[1][1];
		float d = m[2][0];
		float e = m[2][2];

		float ct = m[2][1];
		float st = sqrt(1.0 - ct*ct);

		// rescaled inverse of m:
		// a 0 b   inverse  c*e     0     -b*c
		// 0 c 0     ==>     0  a*e - b*d   0
		// d 0 e           -c*d     0      a*c

		float t0 =  c*e;
		float t1 = -b*c;
		float t2 =  a*e - b*d;
		float t3 = -c*d;
		float t4 =  a*c;

		// T1 = (t3 st - t0 ct) n + t0 v;
		// T2 = t2 Cross[n, v];
		// T3 = (t4 st - t1 ct) n + t1 v;

		// pre-rotate t3 and t4
		t3 = t3*st - t0*ct;
		t4 = t4*st - t1*ct;

		// store the variable terms
		file << t0;
		file << ", ";
		file << t1;
		file << ", ";
		file << t2;
		file << ", ";
		file << t3;
		file << ", ";

		// TEMP: copy into magnitude texture for now
		tabAmplitude[i].x = t4;
	}
	file << "];" << endl;

	file << "var g_ltc_mag_2 = [";
	for (int i = 0; i < N*N; ++i, n += 4)
	{
		file << tabAmplitude[i].x << ", ";
	}
	file << "];" << endl;

	file.close();
}

// export data in C
void writeTabC(mat3 * tab, vec2 * tabAmplitude, int N)
{
	ofstream file("results/ltc.inc");

	file << std::fixed;
	file << std::setprecision(6);

	file << "static const int size = " << N  << ";" << endl << endl;

	file << "static const mat33 tabM[size*size] = {" << endl;
	for(int t = 0 ; t < N ; ++t)
	for(int a = 0 ; a < N ; ++a)
	{
		file << "{";
		file << tab[a + t*N][0][0] << "f, " << tab[a + t*N][0][1] << "f, " << tab[a + t*N][0][2] << "f, ";
		file << tab[a + t*N][1][0] << "f, " << tab[a + t*N][1][1] << "f, " << tab[a + t*N][1][2] << "f, ";
		file << tab[a + t*N][2][0] << "f, " << tab[a + t*N][2][1] << "f, " << tab[a + t*N][2][2] << "f}";
		if(a != N-1 || t != N-1)
			file << ", ";
		file << endl;
	}
	file << "};" << endl << endl;

	file << "static const mat33 tabMinv[size*size] = {" << endl;
	for (int i = 0; i < N*N; ++i)
	{
        const mat3& m = tab[i];

		float a = m[0][0];
		float b = m[0][2];
		float c = m[1][1];
		float d = m[2][0];
		float e = m[2][2];

		float ct = m[2][1];
		float st = sqrt(1.0 - ct*ct);

		// rescaled inverse of m (det(m) = 1):
		// a 0 b   inverse  c*e     0     -b*c
		// 0 c 0     ==>     0  a*e - b*d   0
		// d 0 e           -c*d     0      a*c

		file << "{";
		file <<  c*e << "f, " << "0"       << "f, " << -b*c << "f, ";
		file <<  "0" << "f, " << a*e - b*d << "f, " <<  "0" << "f, ";
		file << -c*d << "f, " << "0"       << "f, " <<  a*c << "f}";
		if(i != (N*N-1))
			file << ", ";
		file << endl;
	}
	file << "};" << endl << endl;

	file << "static const vec2 tabAmplitude[size*size] = {" << endl;
	for(int t = 0 ; t < N ; ++t)
	for(int a = 0 ; a < N ; ++a)
	{
		file << "{";
		file << tabAmplitude[a + t*N][0] << "f, ";
		file << tabAmplitude[a + t*N][1] << "f}";
		if(a != N-1 || t != N-1)
			file << ", ";
		file << endl;
	}
	file << "};" << endl;

	file.close();
}

// export data in matlab
void writeTabMatlab(mat3 * tab, vec2 * tabAmplitude, int N)
{
	ofstream file("results/ltc.mat");

	file << "# name: tabAmplitude" << endl;
	file << "# type: matrix" << endl;
	file << "# ndims: 2" << endl;
	file << " " << N << " " << N << endl;

	for(int t = 0 ; t < N ; ++t)
	{
		for(int a = 0 ; a < N ; ++a)
		{
			file << tabAmplitude[a + t*N][0] << " " ;
			file << tabAmplitude[a + t*N][1] << " " ;
		}
		file << endl;
	}

	for(int row = 0 ; row<3 ; ++row)
	for(int column = 0 ; column<3 ; ++column)
	{

		file << "# name: tab" << column << row << endl;
		file << "# type: matrix" << endl;
		file << "# ndims: 2" << endl;
		file << " " << N << " " << N << endl;

		for(int t = 0 ; t < N ; ++t)
		{
			for(int a = 0 ; a < N ; ++a)
			{
				file << tab[a + t*N][column][row] << " " ;
			}
			file << endl;
		}

		file << endl;
	}

	file.close();
}

// export data in dds
#include "dds.h"

void writeDDS(mat3 * tab, vec2 * tabAmplitude, int N)
{
	assert(false && "This version does not currently support writing the last component of the matrix.");
	float * data = new float[N*N*4];

	int n = 0;
	for (int i = 0; i < N*N; ++i, n += 4)
	{
		const mat3& m = tab[i];

		float a = m[0][0];
		float b = m[0][2];
		float c = m[1][1];
		float d = m[2][0];

		// Rescaled inverse of m:
		// a 0 b   inverse   1      0      -b
		// 0 c 0     ==>     0 (a - b*d)/c  0
		// d 0 1            -d      0       a

		// Store the variable terms
		data[n + 0] =  a;
		data[n + 1] = -b;
		data[n + 2] = (a - b*d) / c;
		data[n + 3] = -d;
	}

	SaveDDS("results/ltc_mat.dds", DDS_FORMAT_R32G32B32A32_FLOAT, sizeof(float)*4, N, N, data);
	SaveDDS("results/ltc_amp.dds", DDS_FORMAT_R32G32_FLOAT,       sizeof(float)*2, N, N, tabAmplitude);

	delete [] data;
}

#endif
