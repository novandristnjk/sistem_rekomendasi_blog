# Laporan Proyek Machine Learning - Novandri Sitinjak

## Domain Proyek
Dalam era digital yang semakin maju, blog telah menjadi sumber informasi yang tak terhitung jumlahnya. Setiap hari, ribuan blog baru muncul di berbagai topik, mulai dari teknologi, gaya hidup, keuangan, hingga kesehatan. Namun, dengan ledakan informasi ini, pengguna sering kali mengalami kesulitan dalam menavigasi melalui banyaknya pilihan dan menemukan blog yang relevan dengan minat mereka.

Sistem rekomendasi blog bertujuan untuk memberikan solusi yang personal dan efektif dalam membantu pengguna menemukan blog yang paling sesuai dengan minat dan preferensi mereka. Dengan menggunakan algoritma dan teknik khusus, sistem ini dapat menyaring dan merekomendasikan blog berdasarkan kebutuhan pengguna.

Salah satu keuntungan utama dari sistem rekomendasi blog adalah penggunaan preferensi individual. Setiap pengguna memiliki minat yang unik dan preferensi yang berbeda dalam hal topik, gaya penulisan, atau pendekatan yang diinginkan. Dengan memanfaatkan data pengguna, seperti riwayat kunjungan, preferensi topik, atau penilaian, sistem rekomendasi dapat menyusun profil pengguna yang akurat dan memberikan rekomendasi yang lebih personal.

Selain itu, sistem rekomendasi blog juga berfokus pada pengalaman pengguna yang unggul. Pengguna saat ini mengharapkan pengalaman yang disesuaikan dengan minat mereka di setiap platform digital yang mereka gunakan. Dengan menggunakan sistem rekomendasi blog, pengguna dapat menemukan blog yang relevan dan menarik dengan lebih mudah, meningkatkan keterlibatan mereka dalam menjelajahi konten dan mengoptimalkan waktu yang dihabiskan di platform.

Sistem rekomendasi blog juga memiliki potensi bisnis yang besar. Dalam lingkungan yang penuh persaingan ini, pemilik blog dan platform berlomba-lomba untuk menarik dan mempertahankan pengguna. Dengan memberikan rekomendasi blog yang paling relevan dan menarik, sistem rekomendasi blog dapat meningkatkan keterlibatan pengguna, memperluas jangkauan audiens, dan meningkatkan kepuasan pengguna. Hal ini berpotensi membawa dampak positif pada pertumbuhan bisnis dan keberhasilan platform.

Dalam rangka menciptakan sistem rekomendasi blog yang efektif, diperlukan pemahaman yang mendalam tentang preferensi pengguna, analisis data yang akurat, dan penggunaan teknik rekomendasi yang canggih. Dengan memanfaatkan kemajuan dalam bidang kecerdasan buatan dan pembelajaran mesin, sistem rekomendasi blog dapat menjadi alat yang berharga bagi pengguna dan pemilik blog untuk mengoptimalkan pengalaman pengguna, menemukan konten yang berkualitas, dan mencapai tujuan bisnis mereka.

## Business Understanding

Berdasarkan uraian diatas, dapat ditarik sebagai berikut

### Problem Statements
1. Bagaimana mengembangkan sistem rekomendasi blog  yang dapat merekomendasikan blog yang relevan dengan presisi 90%?

### Goals
1. Mengembangkan sistem rekomendasi blog  yang dapat merekomendasikan blog yang relevan dengan presisi 90%.

### Solution Statement
   Tahapan untuk menyelesaikan tujuan dari proyek ini adalah sebagai berikut.

* Melakukan Exploratory Data Analysis (EDA) untuk melakukan pembersihan data, visualisasi diagram dan analisis mengenai tren blog.

* Membangun model sistem rekomendasi dimana akan menggunakan pendekatan Content Based Filtering yaitu dengan memberikan rekomendasi berdasarkan Anime yang pernah dibaca pengguna dengan mengukur kesamaan blog berdasarkan topic. Model Content Based Filtering yang akan dibangun menggunaka metode Cosine Similarity dan evaluasi akan menggunakan metode Precission.

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah Blog Recommendation Data yang dari [kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset). Dataset terdiri dari 3 filr csv yaitu

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

Proses Content-Based Filtering dimulai dengan membuat profil atau vektor fitur untuk setiap item dalam dataset. Misalnya, dalam konteks sistem rekomendasi blog, profil konten untuk setiap blog dapat berisi informasi tentang topik, tag, kategori, penulis, dan kata kunci yang terkait dengan blog tersebut.

Selanjutnya, profil konten dari item-item yang relevan dengan preferensi pengguna dikumpulkan untuk membentuk profil preferensi pengguna. Profil ini mencerminkan minat dan preferensi unik pengguna berdasarkan riwayat interaksinya dengan item sebelumnya.

Setelah profil konten dan profil preferensi pengguna terbentuk, metode Content-Based Filtering menggunakan teknik pengukuran kemiripan (seperti cosine similarity atau Jaccard similarity) untuk mencocokkan profil konten item dengan profil preferensi pengguna. Item yang memiliki kemiripan yang tinggi dengan profil preferensi pengguna akan dianggap lebih relevan dan direkomendasikan kepada pengguna.

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
Metrik evaluasi yang digunakan dalam proyek ini adalah presisi

### Presisi(_Precision_)
Presisi menggambarkan sejauh mana model dapat mengidentifikasi dengan benar kasus positif dari prediksi positif yang dilakukan. Metrik ini penting ketika fokus utama adalah mengurangi kesalahan positif palsu (_misclassification_ yang mengatakan bahwa suatu sampel adalah positif ketika sebenarnya negatif).
Presisi berguna dalam kasus di mana kesalahan positif palsu memiliki konsekuensi yang lebih serius, misalnya dalam sistem deteksi penyakit di mana kesalahan diagnosis positif palsu dapat menyebabkan kecemasan yang tidak perlu atau pengobatan yang tidak perlu.

Rumus:

Precision = $\frac{{\text{{True Positive}}}}{{\text{{True Positive}} + \text{{False Positive}}}}$

Pada proyek ini, penulis akan melakukan percobaan untuk mendapatkan rekomendasi blog dimana ada 3 judul blog yang akan dijadikan bahan percobaan. Untuk nilai TP dapat diberikan jika topic tersebut sama dengan blog yang jadi patokan.

Tabel 8. Judul blog yang akan dijadikan patokan untuk mendapatkan daftar rekomendasi blog

|    |   id | title                             | topic               |
|---:|-----:|:----------------------------------|:--------------------|
|  0 | 3024 | Tasha’s Trinkets                  | backend-development |
|  1 | 2226 | How to Emboss Images using Python | image-processing    |
|  2 | 5095 | EVE Ignites NFT Hope              | web3                |

Tabel 9. Daftar rekomendasi blog dengan judul blog patokan Tasha’s Trinkets

|    | title                                                                                                                                                           | topic               |
|---:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------|:--------------------|
|  0 | Build A Signup, Login and Logout Feature                                                                                                                        | backend-development |
|  1 | 5 Python Libraries to Make Backend Development Easy                                                                                                             | backend-development |
|  2 | Building the Backend: A guide on building the backend of your web application using Express.js and MongoDB, including creating routes, controllers, and models. | backend-development |
|  3 | [ AI ] 讓 ChatGPT 當我的 Software Project 同事?                                                                                                                 | backend-development |
|  4 | [ AI ] 讓 ChatGPT 當我的 Software Project 同事?                                                                                                                 | backend-development |

Berdasarkan hasil dari tabel diatas, keseluruhan rekomendasi blog memiliki topic yang sama dengan topic blog patokan. Maka nilai presisi dari rekomendasi pada tabel 9 adalah 100%.

Tabel 10. Daftar rekomendasi blog dengan judul blog patokan How to Emboss Images using Python

|    | title                                                        | topic            |
|---:|:-------------------------------------------------------------|:-----------------|
|  0 | How to Create a Swirl Distortion Effect in PHP               | image-processing |
|  1 | On the Possibility of Turning Image Into Sound               | image-processing |
|  2 | Automating File Type Conversion with Python                  | image-processing |
|  3 | Why do I receive errors using IFANBEAM for matrices with NaN | image-processing |
|  4 | Why do I receive errors using IFANBEAM for matrices with NaN | image-processing |

Berdasarkan hasil dari tabel diatas, keseluruhan rekomendasi blog memiliki topic yang sama dengan topic blog patokan. Maka nilai presisi dari rekomendasi pada tabel 10 adalah 100%.

Tabel 11. Daftar rekomendasi blog dengan judul blog patokan EVE Ignites NFT Hope

|    | title                                                                                           | topic   |
|---:|:------------------------------------------------------------------------------------------------|:--------|
|  0 | The Nibiru Chain #9 Oracle Module is the Next Generation of Decentralized Oracles               | web3    |
|  1 | Smart Contract Security Part III: Advanced Techniques for Unbeatable Security and Peace of Mind | web3    |
|  2 | 2023 Hacking Incident Announcement                                                              | web3    |
|  3 | Meloria -a Solana NFT Project to Keep an Eye On                                                 | web3    |
|  4 | What is Haven1?                                                                                 | web3    |

Dari nilai metrik evaluasi terlihat model memiliki kemampuan rekomendasi yang baik.Content-based filtering yang presisinya bisa mencapai 100% yang artinya tempat yang direkomendasikan sangat relevan dengan tempat yang sedang dicari. Ini juga menjadi kelebihan pendekatan content-based filtering yang bisa secara tepat memberikan rekomendasi barang yang spesifik.


## Conclusion

Berdasarkan hasil dari sistem rekomendasi dengan metode Content-Based Filtering, model dengan metode perhitungan Cosine Similarity memberikan hasil dengan nilai presisi sebesar 100%.

## Reference

[1]   O'Connor, R., & Smyth, B. (2010). From recommender systems to conversational recommendation. In Proceedings of the 4th ACM conference on Recommender systems (pp. 301-304).

[2]   Adomavicius, G., & Tuzhilin, A. (2005). Toward the next generation of recommender systems: a survey of the state-of-the-art and possible extensions. IEEE Transactions on Knowledge and Data Engineering, 17(6), 734-749.

[3]   Ricci, F., Rokach, L., & Shapira, B. (2015). Recommender systems: introduction and challenges. In Recommender Systems Handbook (pp. 1-34). Springer.
