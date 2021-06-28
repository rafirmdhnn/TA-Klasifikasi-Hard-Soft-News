<h1>Implementasi Algoritma Random Forest dan Multinomial Naive Bayes dengan Menggunakan Seleksi Fitur Information Gain untuk Klasifikasi Berita Media Monitoring Kawasan Geopark Ciletuh</h1>
<hr>
<p>Ini merupakan <em>repository</em> yang berisi <em>source code</em> dan <em>dataset</em> yang digunakan untuk keperluan Tugas Akhir dengan judul Implementasi Algoritma Random Forest dan Multinomial Naive Bayes dengan 
Menggunakan Seleksi Fitur Information Gain untuk Klasifikasi Berita Media Monitoring Kawasan Geopark Ciletuh</p>
<hr>
<h2>Abstrak</h2>
<p>Pada era teknologi informasi dan komunikasi seperti saat ini, proses penyebaran informasi menjadi lebih masif dan cepat. 
Hal ini menyebabkan proses media monitoring yang dilakukan praktisi Public Relations (PR) untuk mengembangkan dan mempertahankan reputasi kawasan Geopark Ciletuh menjadi 
kurang efektif dan efisien apabila dalam melakukan identifikasi dan analisis berita masih dilakukan secara manual. Dalam sistem <em>media monitoring</em>, 
proses ini masuk kedalam proses <em>Analysis Backend</em>. Penelitian ini mengusulkan sistem klasifikasi berita dengan menggunakan algoritma <em>machine learning</em> 
untuk membuat proses analisis berita yang dilakukan lebih efektif dan efisien dengan menggunakan dua algoritma <em>machine learning</em> yaitu <em>Random Forest</em> dan 
<em>Multinomial Naive Bayes</em> serta <em>Information Gain</em> sebagai metode pemilihan fitur untuk melakukan klasifikasi berita <em>hard news</em> atau <em>soft news</em>. 
Dataset yang digunakan dibentuk berdasarkan dua model yang berbeda yaitu <em>single dimensional</em> dan <em>multidimensional</em>. Hasil penelitian untuk model 
<em>dataset single dimensional</em> dengan algoritma <em>Random Forest</em> memperoleh rata-rata tertinggi untuk nilai akurasi 72,56% dan 64% untuk <em>f1-score</em> 
dengan menggunakan pemilihan fitur <em>information gain</em>, sedangkan algoritma <em>Multinomial Naive Bayes</em> memperoleh rata-rata tertinggi untuk nilai akurasi 
74,18% dan 75% untuk <em>f1-score</em> tanpa pemilihan fitur <em>information gain</em>. Untuk model <em>dataset multidimensional</em> algoritma <em>Random Forest</em> 
memperoleh rata-rata tertinggi untuk nilai akurasi 96,66%, sedangkan algoritma <em>Multinomial Naive Bayes</em> hanya memperoleh 41%. Dari hasil yang diperoleh, 
penggunaan information gain untuk pemilihan fitur memberikan performa yang kurang baik untuk algoritma <em>Multinomial Naive Bayes</em> karena cara kerja algoritma 
yang menganggap semua fitur bersifat independen dan penggunaan metode <em>laplacian smoothing</em> membuat pengurangan fitur yang dilakukan tidak memberikan hasil yang baik.</p>

<h2>Language and Library</h2>
<ul>
  <li>Python 3.7</li>
  <li>Scikit Learn</li>
  <li>Sastrawi Stemming Library</li>
  <li>NLTK (Natural Language Tools Kit)</li>
  <li>Pandas DataFrame</li>
  <li>NumPy</li>
</ul>

<h2>Documentation</h2>
