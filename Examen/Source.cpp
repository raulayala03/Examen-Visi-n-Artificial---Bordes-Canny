#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <math.h>

using namespace cv;
using namespace std;

int main()
{
	int size, i, j, q, k, l, m;
	float sigma, sum, val;
	float pi = 3.1415926; 

	Mat imagen = imread("lena.png"); // leemos la imagen original 

	cout << "Ingresa el tamaño del kernel (nxn)" << endl;
	cin >> size;
	cout << "Ingresa el valor de sigma" << endl;
	cin >> sigma;

	Mat grises(imagen.rows, imagen.cols, CV_8UC1); // declaramos todas las matrices que se ocuparán 
	cvtColor(imagen, grises, COLOR_RGB2GRAY);
	Mat grisesBord = Mat::zeros(imagen.rows + size - 1, imagen.cols + size - 1, CV_8UC1);
	Mat imgFiltrada = Mat::zeros(imagen.rows, imagen.cols, CV_8UC1);
	Mat imgEcualizada = Mat::zeros(imagen.rows, imagen.cols, CV_8UC1);
	Mat fx = Mat::zeros(imagen.rows, imagen.cols, CV_8UC1);
	Mat fy = Mat::zeros(imagen.rows, imagen.cols, CV_8UC1);
	Mat imgG = Mat::zeros(imagen.rows, imagen.cols, CV_8UC1);
	Mat imgAngle = Mat::zeros(imagen.rows, imagen.cols, CV_8UC1);
	Mat imgNonMax = Mat::zeros(imagen.rows, imagen.cols, CV_8UC1);
	Mat imgStrWk = Mat::zeros(imagen.rows, imagen.cols, CV_8UC1);
	Mat imgFinal = Mat::zeros(imagen.rows, imagen.cols, CV_8UC1);
	Mat kernel = Mat::zeros(size, size, CV_32F);

	sum = 0.0; //kernel
	for (i = 0, k = int(size / 2); k >= -int(size / 2); k--, i++) {
		for (j = 0, q = -int(size / 2); q <= int(size / 2); q++, j++) {
			val = (1 / (2 * 3.1416 * pow(sigma, 2))) * exp((-1 * (pow(q, 2) + pow(k, 2))) / (2 * pow(sigma, 2))); //fórmula del kernel 
			kernel.at<float>(i, j) = val;
			sum = sum + val;
		}
	}
	for (q = 0; q < kernel.rows; q++) //normalizamos el kernel 
		for (k = 0; k < kernel.cols; k++)
			kernel.at<float>(q, k) = kernel.at<float>(q, k) / sum;

	for (i = int(size / 2), q = 0; i < grisesBord.rows - int(size / 2); i++, q++) //imagen con borde
		for (j = int(size / 2), k = 0; j < grisesBord.cols - int(size / 2); j++, k++) //i j inician desde la posición (int(size/2) para agregar borde
			grisesBord.at<uchar>(i, j) = grises.at<uchar>(q, k);

	float aux = 0; //filtro suavizado con kernel (GAUSS)
	for (i = int(size / 2), l = 0; i < grisesBord.rows - int(size / 2); i++, l++) {
		for (j = int(size / 2), m = 0; j < grisesBord.cols - int(size / 2); j++, m++) {
			for (q = 0; q < kernel.rows; q++) {
				for (k = 0; k < kernel.cols; k++) { //recorremos el kernel obteniendo sus valores para aplicar el filtro 
					aux = aux + kernel.at<float>(q, k) * static_cast<float>(grisesBord.at<uchar>(i - int(size / 2) + q, j - int(size / 2) + k));
				}
			}
			imgFiltrada.at<uchar>(l, m) = static_cast<uchar>(aux);
			aux = 0;
		}
	}

	//ecualizado
	cv::equalizeHist(imgFiltrada, imgEcualizada);

	//SOBEL |G|
	Mat gx = (Mat_<float>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1); //declaramos gx y gy
	Mat gy = (Mat_<float>(3, 3) << 1, 2, 1, 0, 0, 0, -1, -2, -1);
	
	for (i = int(size / 2), q = 0; i < grisesBord.rows - int(size / 2); i++, q++) //ecualizada con borde
		for (j = int(size / 2), k = 0; j < grisesBord.cols - int(size / 2); j++, k++)
			grisesBord.at<uchar>(i, j) = imgEcualizada.at<uchar>(q, k); //aplicamos bordes a la img ecualizada para poderle aplicar la derivación 

	float aux2 = 0; //aplicación de |G|
	for (i = int(size / 2), l = 0; i < grisesBord.rows - int(size / 2); i++, l++) {
		for (j = int(size / 2), m = 0; j < grisesBord.cols - int(size / 2); j++, m++) {
			for (q = 0; q < gx.rows; q++) {
				for (k = 0; k < gx.cols; k++) { //obtenemos los valores para fx yfy 
					aux = aux + gx.at<float>(q, k) * static_cast<float>(grisesBord.at<uchar>(i - int(size / 2) + q, j - int(size / 2) + k));
					aux2 = aux2 + gy.at<float>(q, k) * static_cast<float>(grisesBord.at<uchar>(i - int(size / 2) + q, j - int(size / 2) + k));
				}
			}
			fx.at<uchar>(l, m) = static_cast<uchar>(aux);
			fy.at<uchar>(l, m) = static_cast<uchar>(aux2);
			imgG.at<uchar>(l, m) = static_cast<uchar>(sqrt(pow(aux,2) + pow(aux2,2)));
			aux = 0;
			aux2 = 0; 
		}
	}

	//Magnitud y dirección 
	double dir; 
	for (i = 0; i < imgAngle.rows; i++) { 
		for (j = 0; j < imgAngle.cols; j++) { 
			dir = atan(static_cast<double>(fy.at<uchar>(i, j)) / static_cast<double>(fx.at<uchar>(i, j))); //obtenemos la matriz con los ángulos 
			imgAngle.at<uchar>(i, j) = static_cast<uchar>((dir*180)/pi);  //pasamos los radianes a grados        //de dirección 
		}
	}

	//Non Max Supression
	float p, qq, r; 
	
	for (i = 5; i < imgNonMax.rows-1; i++) {
		for (j = 1; j < imgNonMax.cols-1; j++) {
			p = static_cast<float>(imgAngle.at<uchar>(i, j)); 
			qq = 255; 
			r = 255; 

			if ((0 <= p < 22.5) || (157.5 <= p <= 180)) { //ángulo 0
				qq = static_cast<float>(imgG.at<uchar>(i, j + 1));
				r = static_cast<float>(imgG.at<uchar>(i, j - 1));
			}
			else if (22.5 <= p < 67.5) {//ángulo 45
				qq = static_cast<float>(imgG.at<uchar>(i+1, j - 1));
				r = static_cast<float>(imgG.at<uchar>(i-1, j + 1));
			}
			else if (67.5 <= p < 112.5) {//ángulo 90
				qq = static_cast<float>(imgG.at<uchar>(i+1, j));
				r = static_cast<float>(imgG.at<uchar>(i-1, j));
			}
			else if (112.5 <= p < 157.5) {//ángulo 135
				qq = static_cast<float>(imgG.at<uchar>(i-1, j - 1));
				r = static_cast<float>(imgG.at<uchar>(i+1, j + 1));
			}
			
			if ((static_cast<float>(imgG.at<uchar>(i, j) >= qq) && (static_cast<float>(imgG.at<uchar>(i, j) >= r)))) {
				imgNonMax.at<uchar>(i, j) = static_cast<uchar>(imgG.at<uchar>(i,j)); 
			}
			else {
				imgNonMax.at<uchar>(i, j) = static_cast<uchar>(0);
			}
		}
	}

	//Umbrales thresholding
	double minVal, maxVal, highU, lowU;
	Point minLoc, maxLoc; 

	minMaxLoc(imgNonMax, &minVal, &maxVal, &minLoc, &maxLoc);
	highU = maxVal * 0.45; //declaramos el valor para el umbral alto 
	lowU = highU * 0.25; //declaramos el valor para el umbral bajo 
	uchar strong = static_cast<uchar>(255); 
	uchar weak = static_cast<uchar>(25);

	for (i = 3; i < imgNonMax.rows; i++) {
		for (j = 3; j < imgNonMax.cols; j++) {
			if (static_cast<double>(imgNonMax.at<uchar>(i, j)) >= highU) { //si es mayor que el umbral alto es fuerte 
				imgStrWk.at<uchar>(i, j) = strong; 
				imgFinal.at<uchar>(i, j) = strong;
			}
			else if ((static_cast<double>(imgNonMax.at<uchar>(i, j)) <= highU) && (static_cast<double>(imgNonMax.at<uchar>(i, j)) >= lowU)) {
				imgStrWk.at<uchar>(i, j) = weak; //si es menor que el umbral alto y mayor que el umbral bajo es débil  
				imgFinal.at<uchar>(i, j) = weak;
			}
			else if (static_cast<double>(imgNonMax.at<uchar>(i, j)) <= lowU) { //si es menor que el umbral bajo es irrelevante 
				imgStrWk.at<uchar>(i, j) = static_cast<uchar>(0);
				imgFinal.at<uchar>(i, j) = static_cast<uchar>(0);
			}
		}
	}

	////hysteresis
	for (i = 1; i < imgFinal.rows-1; i++) {
		for (j = 1; j < imgFinal.cols-1; j++) {
			if (imgFinal.at<uchar>(i, j) == weak) {
				if ((imgFinal.at<uchar>(i+1, j-1) == strong) || (imgFinal.at<uchar>(i+1, j) == strong) || (imgFinal.at<uchar>(i+1, j+1) == strong)
					|| (imgFinal.at<uchar>(i, j-1) == strong) || (imgFinal.at<uchar>(i, j+1) == strong)
					|| (imgFinal.at<uchar>(i-1, j-1) == strong) || (imgFinal.at<uchar>(i-1, j) == strong) || (imgFinal.at<uchar>(i-1, j+1) == strong)) {
					imgFinal.at<uchar>(i, j) = strong; //si tiene un pixel vecino fuerte entonces se hace fuerte 
				}
				else {
					imgFinal.at<uchar>(i, j) = static_cast<uchar>(0); // de lo contrario se hace irrelevante 
				}
			}
		}
	}

	cout << kernel << endl; //impresión del kernel 
	//impresión de tamaños 
	cout << "Tamaño imagen original: " + to_string(imagen.rows) + 'x' + to_string(imagen.cols) << endl;
	cout << "Tamaño imagen en grises: " + to_string(grises.rows) + 'x' + to_string(grises.cols) << endl;
	cout << "Tamaño imagen con bordes: " + to_string(grisesBord.rows) + 'x' + to_string(grisesBord.cols) << endl;
	cout << "Tamaño imagen con filtro: " + to_string(imgFiltrada.rows) + 'x' + to_string(imgFiltrada.cols) << endl;
	cout << "Tamaño imagen ecualizada: " + to_string(imgEcualizada.rows) + 'x' + to_string(imgEcualizada.cols) << endl;
	cout << "Tamaño imagen |G|: " + to_string(imgG.rows) + 'x' + to_string(imgG.cols) << endl;
	cout << "Tamaño imagen Orientacion de angulos: " + to_string(imgAngle.rows) + 'x' + to_string(imgAngle.cols) << endl;
	cout << "Tamaño imagen Non Max Supress: " + to_string(imgNonMax.rows) + 'x' + to_string(imgNonMax.cols) << endl;
	cout << "Tamaño imagen thresholding: " + to_string(imgStrWk.rows) + 'x' + to_string(imgStrWk.cols) << endl;
	cout << "Tamaño imagen Final (hysteresis): " + to_string(imgFinal.rows) + 'x' + to_string(imgFinal.cols) << endl;
	//impresión de imagenes 
	imshow("Imagen original", imagen);
	imshow("Imagen en escala de grises", grises);
	imshow("Imagen con filtro", imgFiltrada);
	imshow("Imagen ecualizada", imgEcualizada);
	imshow("Imagen |G|", imgG);
	imshow("Imagen Orientacion de angulos", imgAngle);
	imshow("Imagen Non Max supress", imgNonMax);
	imshow("Imagen thresholding", imgStrWk);
	imshow("Imagen Final (hysteresis)", imgFinal);

	waitKey(0);
	return 1;
}