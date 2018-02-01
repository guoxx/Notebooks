#ifndef _LTC_
#define _LTC_


#include "glm\glm.hpp"
using namespace glm;

#include <iostream>
using namespace std;

struct LTC {

	// lobe amplitude
	float amplitude;

	// lobe*fresnel amplitude
	float fresnelAmplitude;

	// parametric representation
	float m11, m22, m13;
	vec3 X, Y, Z;

	// matrix representation
	mat3 M;
	mat3 invM;
	float detM;

	LTC()
	{
		amplitude = 1;
		fresnelAmplitude = 1;
		m11 = 1;
		m22 = 1;
		m13 = 0;
		X = vec3(1,0,0);
		Y = vec3(0,1,0);
		Z = vec3(0,0,1);
		update();
	}

	void update() // compute matrix from parameters
	{
		M = mat3(X, Y, Z) *
			mat3(m11, 0, 0,
				0, m22, 0,
				m13, 0, 1);
		invM = inverse(M);
		detM = abs(glm::determinant(M));
	}

	float eval(const vec3& L) const
	{
		vec3 Loriginal = normalize(invM * L);
		vec3 L_ = M * Loriginal;

		float l = length(L_);
		float Jacobian = detM / (l*l*l);

		float D = 1.0f / 3.14159f * glm::max<float>(0.0f, Loriginal.z); 
		
		float res = amplitude * D / Jacobian;
		return res;
	}

	vec3 sample(const float U1, const float U2) const
	{
		const float theta = acosf(sqrtf(U1));
		const float phi = 2.0f*3.14159f * U2;
		const vec3 L = normalize(M * vec3(sinf(theta)*cosf(phi), sinf(theta)*sinf(phi), cosf(theta)));
		return L;
	}

	void testNormalization() const
	{
		double sum = 0;
		float dtheta = 0.005f;
		float dphi = 0.005f;
		for(float theta = 0.0f ; theta <= 3.14159f ; theta+=dtheta)
		for(float phi = 0.0f ; phi <= 2.0f * 3.14159f ; phi+=dphi)
		{
			vec3 L(cosf(phi)*sinf(theta), sinf(phi)*sinf(theta), cosf(theta));

			sum += sinf(theta) * eval(L);
		}
		sum *= dtheta * dphi;
		cout << "LTC normalization test: " << sum << endl;
		cout << "LTC normalization expected: " << amplitude << endl;
	}
};

#endif
