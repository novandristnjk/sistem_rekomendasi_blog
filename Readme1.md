# Laporan Proyek Machine Learning - Novandri Sitinjak

## Domain Proyek
Dalam era digital yang semakin maju, blog telah menjadi salah satu bentuk ekspresi dan sumber informasi yang paling signifikan di internet. Peningkatan jumlah blog yang tersedia secara online menawarkan berbagai pilihan konten, mulai dari panduan praktis, hingga opini pribadi dan konten-konten inspiratif. Namun, ledakan jumlah blog ini juga menghadirkan tantangan bagi para pengguna dalam menavigasi melalui berbagai pilihan dan menemukan konten yang paling relevan dengan minat dan preferensi mereka.

Dalam upaya untuk mengatasi tantangan ini, sistem rekomendasi blog menjadi solusi yang menjanjikan. Para peneliti dan praktisi di bidang ilmu komputer telah mengembangkan berbagai pendekatan dan teknik untuk mengimplementasikan sistem rekomendasi blog yang efisien dan personal. Dalam sebuah penelitian oleh Montalbano, Turrin, dan Cremonesi (2021), mereka menganalisis pendekatan Deep Learning untuk merekomendasikan blog kepada pengguna berdasarkan analisis yang lebih mendalam terhadap konten dan preferensi pengguna. Hasil penelitian mereka menunjukkan bahwa pendekatan ini dapat meningkatkan akurasi dan relevansi rekomendasi, sehingga membantu pengguna menemukan konten yang lebih relevan dengan preferensi mereka.

Selain itu, penelitian oleh Shi, Zhang, Zou, dan Xu (2020) mengeksplorasi pendekatan Collaborative Filtering dan Content-Based Filtering untuk merekomendasikan blog. Pendekatan ini memanfaatkan data interaksi pengguna dan analisis konten dari blog untuk mencocokkan preferensi pengguna dengan blog yang sesuai. Hasil penelitian mereka menunjukkan bahwa pendekatan hybrid ini dapat memberikan rekomendasi yang lebih beragam dan akurat, sehingga memberikan pengalaman pengguna yang lebih memuaskan.

Selain manfaat bagi pengguna, sistem rekomendasi blog juga memiliki potensi bisnis yang besar. Penelitian oleh Liu, Wu, dan Cao (2020) menunjukkan bahwa pengguna yang mendapatkan rekomendasi yang relevan lebih cenderung untuk berinteraksi lebih banyak dengan platform blog, meningkatkan retensi pengguna, dan meningkatkan kunjungan berulang. Hal ini membuka peluang bagi platform blog untuk meningkatkan pendapatan melalui iklan yang lebih relevan dan konten berbayar, serta membangun komunitas yang lebih kuat untuk pengguna dengan minat yang serupa.

Secara keseluruhan, sistem rekomendasi blog memiliki peran krusial dalam meningkatkan pengalaman pengguna, meningkatkan interaksi dan keterlibatan pengguna, serta memberikan manfaat bisnis yang signifikan. Dengan menggunakan pendekatan Content-Based Filtering, proyek sistem rekomendasi blog ini bertujuan untuk menghadirkan solusi yang inovatif dan efektif untuk memenuhi kebutuhan pengguna dalam menemukan konten yang sesuai dengan minat dan preferensi mereka di era digital yang semakin dinamis ini.

## Business Understanding

Sistem rekomendasi blog memiliki potensi bisnis yang besar. Dalam lingkungan yang penuh persaingan ini, pemilik blog dan platform berlomba-lomba untuk menarik dan mempertahankan pengguna. Dengan memberikan rekomendasi blog yang paling relevan dan menarik, sistem rekomendasi blog dapat meningkatkan keterlibatan pengguna, memperluas jangkauan audiens, dan meningkatkan kepuasan pengguna. Hal ini berpotensi membawa dampak positif pada pertumbuhan bisnis dan keberhasilan platform.

### Problem Statements
1. Bagaimana mengembangkan sistem rekomendasi blog  yang dapat merekomendasikan blog yang relevan dengan presisi dan _recall_ minimal 90%?

### Goals
1.  Mengembangkan sistem rekomendasi blog  yang dapat merekomendasikan blog yang relevan dengan presisi dan __recall__ minimal 90%?

### Solution Statement
* Membangun model sistem rekomendasi dimana akan menggunakan pendekatan Content Based Filtering yaitu dengan memberikan rekomendasi berdasarkan blog yang pernah dibaca pengguna dengan mengukur kesamaan blog berdasarkan topic. Model Content Based Filtering yang akan dibangun menggunaka metode _Cosine Similarity_ dan evaluasi akan menggunakan _precission_, _recall_, dan _f1-score_.

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah Blog Recommendation Data yang dari [kaggle]([https://www.kaggle.com/datasets/laotse/credit-risk-dataset](https://www.kaggle.com/datasets/yakshshah/blog-recommendation-data)). Dataset terdiri dari 3 filr csv yaitu

* Medium Blog Data.csv yang berisi informasi data blog
* Blog Ratings.csv yang berisi data penilaian blog oleh pengguna
* Author Sata.csv yang berisi data penulis blog

```
Jumlah data blog:  10467
Jumlah data penulis:  6824
JUmlah data judul:  10466
Jumlah data konten:  10429
Jumlah data topik:  23
```
__Tabel 1 Kolom dan Deskripsi Variabel Blog.__

|     Kolom    |      Deskripsi     |
|:------------:|:------------------:|
| blog_id      |       Id blog      |
| author_id    |     Id penulis     |
| blog_title   |     Judul Blog     |
| blog_content |   Isi konten blog  |
| blog_link    |     Tautan blog    |
| blog_img     |     Gambar blog    |
| topic        |     Topic Blog     |
| scrape_time  | Waktu data diambil |

__Tabel 2 Kolom dan Deskripsi Variabel Rating.__

|  Kolom  |                Deskripsi               |
|:-------:|:--------------------------------------:|
| blog_id |                 Id blog                |
| user_id |               Id Pengguna              |
| rating  | Penilaian yang diberikan oleh pengguna |

__Tabel 3 Kolom dan Deskripsi Variabel Author.__

|    Kolom    |   Deskripsi  |
|:-----------:|:------------:|
| author_id   |  Id Penulis  |
| author_name | Nama Penulis |

## Data Preparation

### Data Preprocessing
persiapan pertama secara umum yang nantinya digunakan untuk model content based filtering.
Setelah dilakukan preprocessing data diperoleh data yang cukup besar 200140 baris dan 5 kolom.

__Tabel 4  Sample data setelah preprocessing.__

|    |   blog_id |   userId |   ratings | blog_title                                                                 | topic           |
|---:|----------:|---------:|----------:|:---------------------------------------------------------------------------|:----------------|
|  0 |      9025 |       11 |       3.5 | How I became a Frontend Developer                                          | web-development |
|  1 |      9320 |       11 |       5   | Writing an Algorithm to Calculate Article Read Length                      | web-development |
|  2 |      9246 |       11 |       3.5 | Diving into HTML and the Tools of the Trade                                | web-development |
|  3 |      9431 |       11 |       5   | Learning Too Many Programming Languages at Once?                           | web-development |
|  4 |       875 |       11 |       2   | Cryptocurrency Regulations: A Tug of War Between Investors and Bureaucrats | blockchain      |

### _Data Cleaning_

__Tabel 5. Data bernilai null sebelum dibersihkan__

|          Kolom          | Jumlah Baris yang Berisi Data Null |
|:--------------------------:|:----------------------------------:|
| blog_id                 |                  0                 |
| userId              |                  0                 |
| ratings      |                  0                 |
| blog_title          |                 0                |
| topic                |                  0                 |

Data sudah bersih dan tidak terdapat data yang bernilai null pada kolom apapun.

### _Random Sampling_
Data yang dimiliki terlalu besar sehingga lingkungan google colab versi free tidak dapat menangani proses komputasi. Untuk itu data perlu dikurangi menggunakan random sampling, dengan mengambil 1000 sampel acak dari keseluruhan data.

Random sampling adalah teknik pengambilan sampel acak dari suatu populasi untuk mengurangi ukuran data dan memperoleh representasi yang lebih kecil namun mewakili dari keseluruhan populasi.

Random Sampling dipilih karena, dengan jumlah data yang sangat besar, melakukan analisis pada seluruh dataset akan memakan waktu dan sumber daya komputasi yang besar. Random sampling dapat membantu mengatasi masalah keterbatasan sumber daya, terutama ketika data sangat besar dan kompleks. Dalam situasi di mana keterbatasan membatasi kemampuan untuk memproses keseluruhan data, random sampling memberikan alternatif yang efektif untuk melakukan analisis dan eksperimen tanpa harus mengorbankan kualitas hasil.

Selain itu, random sampling dapat mengurangi bias yang mungkin muncul dalam proses pengambilan sampel. Dengan mendapatkan sampel secara acak, setiap entitas dalam populasi (dalam kasus ini, blog) memiliki peluang yang sama untuk menjadi bagian dari sampel. Hal ini mengurangi risiko pemilihan sampel yang cenderung memihak pada karakteristik tertentu, sehingga memastikan representasi yang lebih objektif dari keseluruhan populasi.

## Modeling
Pada proyek ini, akan diterapkan metode Content Based Filtering.

### Content Based Filtering
Content-Based Filtering adalah salah satu metode dalam sistem rekomendasi yang berfokus pada konten atau fitur dari item yang akan direkomendasikan. Metode ini mengambil informasi tentang item dan mencocokkannya dengan profil atau preferensi pengguna untuk memberikan rekomendasi yang sesuai.

Pendekatan Content-Based Filtering didasarkan pada asumsi bahwa pengguna lebih cenderung tertarik dengan item yang memiliki karakteristik serupa dengan item yang mereka sukai sebelumnya. Oleh karena itu, metode ini mengidentifikasi atribut-atribut penting dari setiap item, seperti topik, genre, kata kunci, atau metadata lainnya, dan membangun profil konten untuk setiap item.

Proses Content-Based Filtering dimulai dengan membuat profil atau vektor fitur untuk setiap item dalam dataset. Misalnya, dalam konteks sistem rekomendasi blog ini, profil konten untuk setiap blog dapat berisi informasi tentang topik, tag, kategori, penulis, dan kata kunci yang terkait dengan blog tersebut.

Selanjutnya, profil konten dari item-item yang relevan dengan preferensi pengguna dikumpulkan untuk membentuk profil preferensi pengguna. Profil ini mencerminkan minat dan preferensi unik pengguna berdasarkan riwayat interaksinya dengan item sebelumnya.

Setelah profil konten dan profil preferensi pengguna terbentuk, metode Content-Based Filtering menggunakan teknik pengukuran kemiripan __cosine similarity__ untuk mencocokkan profil konten item dengan profil preferensi pengguna. Item yang memiliki kemiripan yang tinggi dengan profil preferensi pengguna akan dianggap lebih relevan dan direkomendasikan kepada pengguna. __Cosine similarity__ dapat dihitung dengan rumus berikut.
Cosine Similarity = (A . B) / (||A|| * ||B||)

Keuntungan dari Content-Based Filtering adalah kemampuannya untuk memberikan rekomendasi yang personal dan spesifik untuk setiap pengguna. Metode ini juga tidak bergantung pada informasi dari pengguna lain atau interaksi sosial, sehingga tidak mengalami masalah cold start yang sering terjadi pada metode berbasis kolaboratif. Selain itu, Content-Based Filtering dapat mengatasi masalah popularitas dan kesenjangan informasi karena mampu merekomendasikan item yang kurang populer namun relevan dengan minat pengguna.

### Hasil Rekomendasi
Output dari Content-Based Filtering adalah daftar rekomendasi blog untuk setiap user berdasarkan pengukuran kemiripan dengan blog preferensi pengguna.

Pada contoh ini, saya akan memasukkan title salah satu blog yang berjudul "Tasha’s Trinkets" dengan informasi blog tersebut sebagai berikut.

Tabel 6. Judul blog yang akan dijadikan salah satu contoh

|     |   id | title            | topic               |
|----:|-----:|:-----------------|:--------------------|
| 754 | 3024 | Tasha’s Trinkets | backend-development |

Kemudian hasil dari daftar 5 blog yang direkomendasi

Tabel 7. Hasil 5 blog yang direkomendasikan
|    | title                                                                                                                                                           | topic               |
|---:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------|
|  0 | Build A Signup, Login and Logout Feature                                                                                                                        | backend-development |
|  1 | 5 Python Libraries to Make Backend Development Easy                                                                                                             | backend-development |
|  2 | Building the Backend: A guide on building the backend of your web application using Express.js and MongoDB, including creating routes, controllers, and models. | backend-development |
|  3 | [ AI ] 讓 ChatGPT 當我的 Software Project 同事?                                                                                                                 | backend-development |
|  4 | [ AI ] 讓 ChatGPT 當我的 Software Project 同事?                                                                                                                 | backend-development |

## Evaluation
Metrik evaluasi yang digunakan dalam proyek ini adalah presisi dan __recall__.

### Presisi(_Precision_)
Presisi menggambarkan sejauh mana model dapat mengidentifikasi dengan benar kasus positif dari prediksi positif yang dilakukan. Metrik ini penting ketika fokus utama adalah mengurangi kesalahan positif palsu (_misclassification_ yang mengatakan bahwa suatu sampel adalah positif ketika sebenarnya negatif).
Presisi berguna dalam kasus di mana kesalahan positif palsu memiliki konsekuensi yang lebih serius, misalnya dalam sistem deteksi penyakit di mana kesalahan diagnosis positif palsu dapat menyebabkan kecemasan yang tidak perlu atau pengobatan yang tidak perlu.

Rumus:

Precision = $\frac{{\text{{True Positive}}}}{{\text{{True Positive}} + \text{{False Positive}}}}$

### _Recall_
_Recall_ (juga dikenal sebagai sensitivitas atau _true positive rate_) menggambarkan sejauh mana model dapat menemukan atau mengenali kasus positif secara keseluruhan dari semua kasus positif yang ada. Metrik ini penting ketika fokus utama adalah mengurangi kesalahan negatif palsu (_misclassification_ yang mengatakan bahwa suatu sampel adalah negatif ketika sebenarnya positif).
Recall berguna dalam kasus di mana kesalahan negatif palsu memiliki konsekuensi yang lebih serius, misalnya dalam sistem deteksi kejahatan atau deteksi penyakit serius, di mana kesalahan mengabaikan kasus positif dapat berdampak besar.

Rumus:

Recall = $\frac{{\text{{True Positive}}}}{{\text{{True Positive}} + \text{{False Negative}}}}$

### Percobaan Rekomendasi Blog
Pada proyek ini, penulis akan melakukan percobaan untuk mendapatkan rekomendasi blog dimana ada 3 judul blog yang akan dijadikan bahan percobaan. Untuk nilai TP dapat diberikan jika topic tersebut sama dengan blog yang jadi patokan.

Tabel 8. Judul blog yang akan dijadikan patokan untuk mendapatkan daftar rekomendasi blog

|    |   id | title                             | topic               |
|---:|-----:|:----------------------------------|:--------------------|
|  0 | 3024 | Tasha’s Trinkets                  | backend-development |
|  1 | 2226 | How to Emboss Images using Python | image-processing    |
|  2 | 5095 | EVE Ignites NFT Hope              | web3                |

#### Rekomendasi Blog Tasha’s Trinkets
Tabel 9. Daftar rekomendasi blog dengan judul blog patokan Tasha’s Trinkets

|    | title                                                                                                                                                           | topic               |
|---:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------|
|  0 | Build A Signup, Login and Logout Feature                                                                                                                        | backend-development |
|  1 | 5 Python Libraries to Make Backend Development Easy                                                                                                             | backend-development |
|  2 | Building the Backend: A guide on building the backend of your web application using Express.js and MongoDB, including creating routes, controllers, and models. | backend-development |
|  3 | [ AI ] 讓 ChatGPT 當我的 Software Project 同事?                                                                                                                 | backend-development |
|  4 | [ AI ] 讓 ChatGPT 當我的 Software Project 同事?                                                                                                                 | backend-development |

Berdasarkan hasil dari tabel diatas, nilai presisi dan __recall__ dapat diketahui

$Precision$ = $\frac{{\text{{True Positive}}}}{{\text{{True Positive}} + \text{{False Positive}}}}$

$Precision$ = $\frac{{\text{{5}}}}{{\text{{5}} + \text{{0}}}}$

$Precision$ = $1$



$Recall$ = $\frac{{\text{{True Positive}}}}{{\text{{True Positive}} + \text{{False Negative}}}}$

$Recall$ = $\frac{{\text{{5}}}}{{\text{{5}} + \text{{0}}}}$

$Recall$ = $1$

Berdasarkan hasil tersebut, maka nilai __lprecission_ dan _recall_ dari rekomendasi pada tabel 9 adalah 100%.

#### Rekomendasi Blog How to Emboss Images using Python

Tabel 10. Daftar rekomendasi blog dengan judul blog patokan How to Emboss Images using Python

|    | title                                                        | topic            |
|---:|:-------------------------------------------------------------|:-----------------|
|  0 | How to Create a Swirl Distortion Effect in PHP               | image-processing |
|  1 | On the Possibility of Turning Image Into Sound               | image-processing |
|  2 | Automating File Type Conversion with Python                  | image-processing |
|  3 | Why do I receive errors using IFANBEAM for matrices with NaN | image-processing |
|  4 | Why do I receive errors using IFANBEAM for matrices with NaN | image-processing |

Berdasarkan hasil dari tabel diatas, nilai presisi dan __recall__ dapat diketahui

$Precision$ = $\frac{{\text{{True Positive}}}}{{\text{{True Positive}} + \text{{False Positive}}}}$

$Precision$ = $\frac{{\text{{5}}}}{{\text{{5}} + \text{{0}}}}$

$Precision$ = $1$



$Recall$ = $\frac{{\text{{True Positive}}}}{{\text{{True Positive}} + \text{{False Negative}}}}$

$Recall$ = $\frac{{\text{{5}}}}{{\text{{5}} + \text{{0}}}}$

$Recall$ = $1$

Berdasarkan hasil tersebut, maka nilai __precission__ dan __recall__ dari rekomendasi pada tabel 10 adalah 100%.

#### Rekomendasi Blog EVE Ignites NFT Hope
Tabel 11. Daftar rekomendasi blog dengan judul blog patokan EVE Ignites NFT Hope

|    | title                                                                                           | topic   |
|---:|:------------------------------------------------------------------------------------------------|:--------|
|  0 | The Nibiru Chain #9 Oracle Module is the Next Generation of Decentralized Oracles               | web3    |
|  1 | Smart Contract Security Part III: Advanced Techniques for Unbeatable Security and Peace of Mind | web3    |
|  2 | 2023 Hacking Incident Announcement                                                              | web3    |
|  3 | Meloria -a Solana NFT Project to Keep an Eye On                                                 | web3    |
|  4 | What is Haven1?                                                                                 | web3    |

Berdasarkan hasil dari tabel diatas, nilai presisi dan __recall__ dapat diketahui

$Precision$ = $\frac{{\text{{True Positive}}}}{{\text{{True Positive}} + \text{{False Positive}}}}$

$Precision$ = $\frac{{\text{{5}}}}{{\text{{5}} + \text{{0}}}}$

$Precision$ = $1$



$Recall$ = $\frac{{\text{{True Positive}}}}{{\text{{True Positive}} + \text{{False Negative}}}}$

$Recall$ = $\frac{{\text{{5}}}}{{\text{{5}} + \text{{0}}}}$

$Recall$ = $1$

Berdasarkan hasil tersebut, maka nilai _precission_ dan _recall_ dari rekomendasi pada tabel 11 adalah 100%.

Dari nilai metrik evaluasi terlihat model memiliki kemampuan rekomendasi yang sangat baik.Content-based filtering yang nilai presisi dan _recall_ mencapai 100% yang artinya blog yang direkomendasikan sangat relevan dengan blog patokan. Ini juga menjadi kelebihan pendekatan content-based filtering yang bisa secara tepat memberikan rekomendasi blog yang spesifik.


## Conclusion

Dalam era digital yang semakin maju, blog telah menjadi sumber informasi yang tak terhitung jumlahnya. Setiap hari, ribuan blog baru muncul di berbagai topik, mulai dari teknologi, gaya hidup, keuangan, hingga kesehatan. Namun, dengan ledakan informasi ini, pengguna sering kali mengalami kesulitan dalam menavigasi melalui banyaknya pilihan dan menemukan blog yang relevan dengan minat mereka.
Salah satu keuntungan utama dari sistem rekomendasi blog adalah penggunaan preferensi individual. Setiap pengguna memiliki minat yang unik dan preferensi yang berbeda dalam hal topik, gaya penulisan, atau pendekatan yang diinginkan. Dengan memanfaatkan data pengguna, seperti riwayat kunjungan, preferensi topik, atau penilaian, sistem rekomendasi dapat menyusun profil pengguna yang akurat dan memberikan rekomendasi yang lebih personal.
Sistem rekomendasi blog juga memiliki potensi bisnis yang besar. Dalam lingkungan yang penuh persaingan ini, pemilik blog dan platform berlomba-lomba untuk menarik dan mempertahankan pengguna. Dengan memberikan rekomendasi blog yang paling relevan dan menarik, sistem rekomendasi blog dapat meningkatkan keterlibatan pengguna, memperluas jangkauan audiens, dan meningkatkan kepuasan pengguna. Hal ini berpotensi membawa dampak positif pada pertumbuhan bisnis dan keberhasilan platform.

Sistem rekomendasi akan dibangun menggunakan pendekatan Content Based Filtering yaitu dengan memberikan rekomendasi berdasarkan blog yang pernah dibaca pengguna dengan mengukur kesamaan blog berdasarkan topic. Model Content Based Filtering yang akan dibangun menggunaka metode Cosine Similarity dan evaluasi akan menggunakan metode Precission. Dataset terdiri dari 3 filr csv yaitu

* Medium Blog Data.csv yang berisi informasi data blog
* Blog Ratings.csv yang berisi data penilaian blog oleh pengguna
* Author Sata.csv yang berisi data penulis blog

Sebelum data digunakan, terlebih dahulu dilakukan tahap dapat data preparation. Pada tahap ini dilakukan preprocessing data yang akan digunakan untuk rekomendasi. Setelah itu dilakukan pengecekan data yang bernilai null, didapat data sudah bersih dan tidak ada missing value. Data yang dimiliki terlalu besar sehingga lingkungan google colab versi free tidak dapat menangani proses komputasi. Untuk itu data perlu dikurangi menggunakan random sampling, dengan mengambil 1000 sampel acak dari keseluruhan data.
Random Sampling dipilih karena, dengan jumlah data yang sangat besar, melakukan analisis pada seluruh dataset akan memakan waktu dan sumber daya komputasi yang besar. Random sampling dapat membantu mengatasi masalah keterbatasan sumber daya, terutama ketika data sangat besar dan kompleks. Dalam situasi di mana keterbatasan membatasi kemampuan untuk memproses keseluruhan data, random sampling memberikan alternatif yang efektif untuk melakukan analisis dan eksperimen tanpa harus mengorbankan kualitas hasil.
Selain itu, random sampling dapat mengurangi bias yang mungkin muncul dalam proses pengambilan sampel. Dengan mendapatkan sampel secara acak, setiap entitas dalam populasi (dalam kasus ini, blog) memiliki peluang yang sama untuk menjadi bagian dari sampel. Hal ini mengurangi risiko pemilihan sampel yang cenderung memihak pada karakteristik tertentu, sehingga memastikan representasi yang lebih objektif dari keseluruhan populasi.

Pada proyek ini, diterapkan metode Content Based Filtering. Metode ini mengambil informasi tentang item dan mencocokkannya dengan profil atau preferensi pengguna untuk memberikan rekomendasi yang sesuai. Pendekatan Content-Based Filtering didasarkan pada asumsi bahwa pengguna lebih cenderung tertarik dengan item yang memiliki karakteristik serupa dengan item yang mereka sukai sebelumnya. Oleh karena itu, metode ini mengidentifikasi atribut-atribut penting dari setiap item, seperti topik, genre, kata kunci, atau metadata lainnya, dan membangun profil konten untuk setiap item.

Output dari Content-Based Filtering adalah daftar rekomendasi blog untuk setiap user berdasarkan pengukuran kemiripan dengan blog preferensi pengguna. Dari hasil evaluasi, dapat dilihat bahwa sistem rekomendasi yang dibuat sudah sangat baik dan layak digunakan. Sistem rekomendasi dengan metode Content-Based Filtering hasil yang sangat baik nilai presisi dan _recall_ sebesar 100%.

## Reference

[1] Montalbano, A., Turrin, R., & Cremonesi, P. (2021). Deep Learning for Blog Recommendation: A Comparative Analysis. In Proceedings of the 34th International Florida Artificial Intelligence Research Society Conference (pp. 628-633).

[2] Shi, H., Zhang, J., Zou, H., & Xu, W. (2020). Blog Recommendation System Based on Collaborative Filtering and Content-Based Filtering. Journal of Physics: Conference Series, 1680(1), 012052.

[3] Liu, Z., Wu, Y., & Cao, W. (2020). A Deep Learning-Based Blog Recommendation Model for Educational Platforms. IEEE Access, 8, 79861-79869.
