# 2. So sánh Quantum Random Walk và Classical Random Walk

## 2.1. Classical random walk làm baseline

Xét simple symmetric random walk

$$
X_T=\sum_{t=1}^{T}\xi_t,\qquad
\Pr(\xi_t=1)=\Pr(\xi_t=-1)=\frac12,
$$

với các increment độc lập. Khi $X_0=0$,

$$
\mathbb E[X_T]=0,\qquad
\operatorname{Var}(X_T)=T,\qquad
\sigma_{\rm CRW}(T)=\sqrt T.
$$

Phân phối chính xác là binomial trên các vị trí có cùng parity với $T$:

$$
\Pr(X_T=x)=
2^{-T}\binom{T}{(T+x)/2},
$$

nếu $|x|\le T$ và $T+x$ chẵn; ngược lại xác suất bằng zero. Sau chuẩn hóa
$X_T/\sqrt T$, central limit theorem cho giới hạn Gaussian. Đây là diffusive scaling:
độ rộng đặc trưng tăng như $O(\sqrt T)$.

## 2.2. QRW lan truyền ballistic

Với DTQRW Hadamard coherent, evolution là $|\psi_T\rangle=U^T|\psi_0\rangle$.
Fourier decomposition biến $U$ thành các block $U(k)$ có eigenphase
$\omega_j(k)$. Thành phần ở vị trí $x$ là tích phân dao động theo pha
$kx-T\omega_j(k)$. Những điểm stationary thỏa

$$
\frac{x}{T}=\frac{d\omega_j(k)}{dk}
$$

đóng góp chủ đạo. Do $x/T$ hội tụ tới một biến vận tốc không suy biến, vị trí điển hình
có độ lớn tỷ lệ trực tiếp với $T$. Vì vậy

$$
\sigma_{\rm QRW}(T)=O(T),\qquad
\operatorname{Var}(X_T)=O(T^2).
$$

Đối với Hadamard walk với initial coin đối xứng theo convention của dự án, định lý giới
hạn Konno cho hệ số bậc hai

$$
\lim_{T\to\infty}\frac{\operatorname{Var}(X_T)}{T^2}
=1-\frac{1}{\sqrt2}
\approx0.292893.
$$

Do đó

$$
\operatorname{Var}(X_T)
=\left(1-\frac1{\sqrt2}\right)T^2+O(T).
$$

Điều này sửa một công thức trong kế hoạch ban đầu ghi
$T^2/2-T/4+O(1)$. Hệ số $1/2$ không đúng cho Hadamard walk chuẩn đang dùng. Notebook
Phase 1 kiểm tra trực tiếp hệ số số học thay vì mã hóa công thức sai vào test.

## 2.3. Hình dạng phân phối

CRW có khối xác suất lớn quanh gốc và tiến dần tới Gaussian khi scale bởi $\sqrt T$.
Hadamard QRW coherent thường có:

- hai front/peak nổi bật gần $x\approx\pm T/\sqrt2$;
- cấu trúc dao động do giao thoa ở miền bên trong;
- xác suất rất nhỏ ngoài "light cone" $|x|\le T$;
- support chỉ trên parity class phù hợp;
- dip tương đối quanh tâm so với CRW, tùy initial coin.

Nói "peak đúng tại $\pm T/\sqrt2$" chỉ là mô tả tiệm cận. Ở finite $T$, vị trí cực đại
bị ảnh hưởng bởi parity, oscillation và initial phase. Đại lượng ổn định hơn để so sánh
là variance scaling, quantile của $X_T/T$, hoặc toàn bộ Wasserstein distance.

## 2.4. Bảng so sánh

| Thuộc tính | CRW đối xứng | DTQRW Hadamard coherent |
|---|---:|---:|
| Trạng thái tiến hóa | Xác suất không âm | Amplitude phức |
| Quy tắc cộng đường đi | Cộng xác suất | Cộng amplitude rồi bình phương |
| Evolution | Stochastic/Markov | Unitary trước phép đo |
| Độ lệch chuẩn | $\sqrt T$ | $T$ |
| Phương sai | $T$ | $\approx(1-1/\sqrt2)T^2$ |
| Scale giới hạn | $X_T/\sqrt T$ | $X_T/T$ |
| Hình dạng điển hình | Gaussian một đỉnh | Hai front và oscillation |
| Phụ thuộc pha đầu | Không | Có |
| Ảnh hưởng đo mỗi bước | Chính là dynamics | Phá coherence, đưa về Markov |
| Memory hiệu dụng | Không với increment iid | Giao thoa giữ thông tin pha của lịch sử |

## 2.5. Ảnh hưởng của coin operator

### Hadamard coin

$$
H=\frac1{\sqrt2}
\begin{pmatrix}1&1\\1&-1\end{pmatrix}.
$$

Hadamard cân bằng về magnitude nhưng không bảo đảm phân phối vị trí đối xứng cho mọi
initial coin. Nó là baseline tốt vì có kết quả giải tích và literature phong phú.

### Biased rotation/reflection coin

Với

$$
C(\theta)=
\begin{pmatrix}
\cos\theta&\sin\theta\\
\sin\theta&-\cos\theta
\end{pmatrix},
$$

$\theta$ thay đổi transition magnitude và miền group velocity. Coin không đối xứng có
thể tạo drift hoặc thay đổi độ rộng, nhưng drift còn phụ thuộc initial phase. Vì thế
không thể diễn giải $\theta$ như một xác suất buy đơn lẻ mà bỏ qua state hiện tại.

### Grover coin

Grover diffusion coin trong chiều $d$ là $G_d=2|s\rangle\langle s|-I_d$. Với coin hai
chiều,

$$
G_2=\begin{pmatrix}0&1\\1&0\end{pmatrix},
$$

chỉ là swap (Pauli $X$). Nó có động lực đặc biệt và không tự động tạo "localization"
trên đường thẳng một chiều. Localization thường được thảo luận rõ hơn với Grover walk
trên lattice hai chiều, graph bậc lớn, defect hoặc disorder. Do đó coin này không nên
được gắn nhãn mean-reversion/localization nếu chưa có kiểm chứng cho graph cụ thể.

### Fourier coin

Discrete Fourier coin thực sự khác Hadamard khi dimension coin lớn hơn hai. Trong hai
chiều, Fourier matrix

$$
F_2=\frac1{\sqrt2}
\begin{pmatrix}1&1\\1&-1\end{pmatrix}=H.
$$

Vì vậy danh sách "Hadamard, Grover, Fourier" không tạo ba baseline độc lập cho walk một
chiều coin hai trạng thái. Để nghiên cứu coin nhiều chiều cần graph hoặc step set nhiều
hướng hơn.

## 2.6. Vai trò của initial state

Với initial coin $|\uparrow\rangle$, Hadamard walk thường lệch về một phía. Với

$$
|\chi_{\rm sym}\rangle
=\frac{|\uparrow\rangle+i|\downarrow\rangle}{\sqrt2},
$$

cross term do relative phase cân bằng hai phía theo convention hiện tại. Tổng quát,

$$
|\chi\rangle=
\cos(\vartheta/2)|\uparrow\rangle+
e^{i\varphi}\sin(\vartheta/2)|\downarrow\rangle.
$$

Hai tham số Bloch-sphere $(\vartheta,\varphi)$ ảnh hưởng mean, skewness và oscillation.
Khi calibrate market model, coin parameter và initial state có thể bù trừ nhau; đây là
vấn đề identifiability. Một protocol hợp lý là cố định initial state đối xứng khi so
variance, rồi chỉ mở parameter initial state trong phân tích drift/skew.

## 2.7. Decoherence và crossover sang diffusion

Nếu sau mỗi bước ta đo đầy đủ coin-position rồi bỏ kết quả, density matrix mất
off-diagonal coherence. Với Hadamard, transition probabilities đều bằng $1/2$, do đó
diagonal distribution tiến hóa như CRW. Với decoherence một phần, thường có crossover:

$$
\operatorname{Var}(X_T)
\sim
\begin{cases}
cT^2,&T\ll\tau_{\rm dec},\\
D(\eta)T,&T\gg\tau_{\rm dec},
\end{cases}
$$

trong đó $\tau_{\rm dec}$ phụ thuộc channel và cường độ noise. Romanelli và cộng sự cho
thấy measurement hoặc broken links có thể làm mất tăng trưởng bậc hai và đưa walk về
diffusive behavior. Tuy nhiên coefficient $D$ không nhất thiết bằng CRW chuẩn trong mọi
channel.

## 2.8. Kiểm chứng hữu hạn trong notebook

Notebook thực hiện bốn lớp kiểm tra:

1. Tính exact endpoint distribution từ statevector, không dùng sampling để tiến hóa.
2. Kiểm tra tổng xác suất và symmetry tới sai số floating-point.
3. So sánh $\operatorname{Var}_{QRW}(T)$ với $T$ của CRW cho $T\le100$.
4. Sample 1000 endpoint measurements để minh họa sai số Monte Carlo, seed cố định.

Tại $T=100$, QRW có variance cỡ $0.2929T^2$, còn CRW có variance $T$. Tỷ lệ lý thuyết
xấp xỉ $29.3$, vượt xa checkpoint $1.5$. Tỷ lệ này không có nghĩa QRW "tốt hơn" trong
dự báo; nó chỉ xác nhận hai dynamics thuộc hai scaling regime khác nhau.

## 2.9. Hàm ý cho dữ liệu thị trường

Giá thực không thể lan truyền ballistic vô hạn vì tick size, liquidity, mean reversion,
volatility clustering và trading halts. Một QRW thuần coherent vì vậy là null model,
không phải mô tả thực nghiệm mặc định. Câu hỏi khoa học hợp lý là liệu một khoảng scale
ngắn có exponent variance lớn hơn một, hoặc một QRW có decoherence/adaptive coin có tái
tạo phân phối tốt hơn baseline hay không. Nếu calibration đẩy decoherence lên cao và
scaling về $T$, đó là kết quả hợp lệ cho thấy coherence analogy không cần thiết ở scale
đang xét.

## 2.10. Kết luận

Khác biệt cốt lõi là QRW cộng amplitude có pha, còn CRW cộng xác suất. Hệ quả là
ballistic spreading so với diffusive spreading. Hệ số variance phụ thuộc coin và
initial state; với baseline Hadamard đối xứng của dự án, hệ số đúng là
$1-1/\sqrt2$, không phải $1/2$. Coin labels cũng phải được dùng thận trọng: Fourier
coin hai chiều chính là Hadamard, còn Grover $2\times2$ chỉ là swap. Các chi tiết này
quan trọng vì một implementation vẫn có thể normalize hoàn hảo nhưng kiểm tra sai một
lý thuyết được phát biểu không chính xác.

