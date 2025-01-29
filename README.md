# CrowdCounting
## Overview
Proyek ini bertujuan untuk mengembangkan sistem deteksi individu dalam kerumunan dengan fokus pada identifikasi bagian kepala sebagai indikator utama. Sistem ini dirancang untuk mendukung analisis kepadatan dan pergerakan massa di berbagai lingkungan, seperti area publik, acara besar, atau fasilitas transportasi. Dengan memanfaatkan teknologi deteksi objek yang akurat, sistem ini diharapkan dapat membantu dalam pemantauan kerumunan, pengelolaan keamanan, serta pengambilan keputusan yang lebih efektif dalam situasi yang memerlukan pengendalian jumlah orang di suatu area.

## Dataset
Dataset didapatkan dari gabungan dataset [crowd counting dataset Computer Vision Project](https://universe.roboflow.com/crowd-dataset/crowd-counting-dataset-w3o7w/dataset/2) yang memiliki gambar kerumunan atau keramaian yang bervariasi. Persebaran data yang digunakan adalah sebagai berikut:  
| Folder  | Image Count | 
| ------------- | ------------- |
| Train  | 2285 | 
| Validation | 382 | 
| Test | 231 |

## Model 
Proyek ini menggunakan model YOLOv8 sebagai algoritma deteksi objek utama, yang telah diimplementasikan dengan SORT (Simple, Online and Realtime Tracker) untuk mencapai berbagai use case seperti line counting, zone counting, dan in/out counting.

### Environment
- GeForce RTX 4060
- Python 3.12.1
- Pytorch 2.5.1
- Torchvision 0.20.1
- Torchaudio 2.5.1
- Ultralytics 8.3.39
- pandas 2.1.4
- opencv 4.10.0.84

### Metrik Evaluasi
![results](https://github.com/user-attachments/assets/7d39848d-39c3-46de-91d5-bae8ca330396)

| Model | epoch  | Imgsz | lr0  | lrf | Recall  | Precision | mAP50  | mAP50-95 |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Baseline | 100  | 640  | 0.01  | 0.01 | 0.71812  | 0.40002  | 0.49617 | 0.21107  |

### Hasil
#### Dwell Time
Mengukur durasi waktu sebuah objek yang terdeteksi sejak pertama kali muncul pada video.
![image](https://github.com/user-attachments/assets/68e6e3f8-6358-4f18-8015-13835a3399fc)

#### Line Counting
Menghitung jumlah objek yang berada dalam sebuah area yang didefinisikan.
![image](https://github.com/user-attachments/assets/468e7625-030e-4857-b35c-22d385899796)

#### Zone & In/Out Counting
Menghitung jumlah objek yang masuk dan keluar dari area tertentu.
![image (1)](https://github.com/user-attachments/assets/c61d99a2-b669-44eb-8027-b7d3e84081fa)

#### Integrasi MySQL
Data hasil deteksi seperti session, personID, koordinat bounding box (x1,y1,w,h), dan Dwell Time disimpan ke dalam database MySQL.
![image (2)](https://github.com/user-attachments/assets/ac9b4176-d764-4922-8aec-65036b8b5741)



