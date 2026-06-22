# 1. DTQRW trên đường thẳng: formalism và quy ước

## 1.1. Phạm vi

Ghi chú này xây dựng discrete-time coined quantum random walk (DTQRW) một chiều trên
đường thẳng vô hạn. Mục tiêu là cố định một hệ quy ước toán học dùng xuyên suốt dự án:
coin state `up` dịch sang phải, coin state `down` dịch sang trái, coin được áp dụng trước
conditional shift, và phép đo vị trí chỉ được thực hiện sau bước tiến hóa cuối cùng.

Tên "random walk" dễ gây hiểu nhầm. Giữa hai phép đo, DTQRW là một tiến hóa tuyến tính,
xác định và unitary của biên độ phức. Tính ngẫu nhiên xuất hiện ở kết quả đo theo Born
rule. Khác biệt này là nguồn gốc của giao thoa và tốc độ lan truyền khác classical random
walk (CRW).

## 1.2. Không gian Hilbert và trạng thái

Không gian coin là

$$
\mathcal H_C = \mathbb C^2
= \operatorname{span}\{|\uparrow\rangle,|\downarrow\rangle\},
$$

và không gian vị trí trên lattice nguyên là

$$
\mathcal H_P = \ell^2(\mathbb Z)
= \operatorname{span}\{|x\rangle:x\in\mathbb Z\}.
$$

Không gian toàn phần là tensor product

$$
\mathcal H=\mathcal H_C\otimes\mathcal H_P.
$$

Một trạng thái tổng quát tại thời điểm nguyên $t$ được viết

$$
|\psi_t\rangle =
\sum_{x\in\mathbb Z}
\left(
  a_x(t)|\uparrow,x\rangle+
  b_x(t)|\downarrow,x\rangle
\right),
$$

trong đó $a_x(t),b_x(t)\in\mathbb C$. Điều kiện chuẩn hóa là

$$
\langle\psi_t|\psi_t\rangle
=\sum_x\left(|a_x(t)|^2+|b_x(t)|^2\right)=1.
$$

Xác suất đo walker tại vị trí $x$ sau $t$ bước là

$$
P(x,t)=|a_x(t)|^2+|b_x(t)|^2.
$$

Hai biên độ coin tại cùng một vị trí được cộng theo xác suất, nhưng các đường đi dẫn
đến cùng biên độ được cộng ở cấp amplitude trước khi lấy bình phương. Vì amplitude có
pha, chúng có thể tăng cường hoặc triệt tiêu nhau.

## 1.3. Coin operator

Coin là một toán tử unitary $C:\mathcal H_C\to\mathcal H_C$. Coin chuẩn của dự án là
Hadamard:

$$
H=\frac{1}{\sqrt 2}
\begin{pmatrix}
1&1\\
1&-1
\end{pmatrix}.
$$

Ta có

$$
H^\dagger H
=\frac12
\begin{pmatrix}
1&1\\
1&-1
\end{pmatrix}
\begin{pmatrix}
1&1\\
1&-1
\end{pmatrix}
=I_2.
$$

Một họ coin thực thường dùng để điều khiển bias là

$$
C(\theta)=
\begin{pmatrix}
\cos\theta&\sin\theta\\
\sin\theta&-\cos\theta
\end{pmatrix}.
$$

$C(\pi/4)=H$. Đây là reflection-rotation coin có định thức $-1$. Nếu cần rotation có
định thức $+1$, có thể dùng

$$
R(\theta)=
\begin{pmatrix}
\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta
\end{pmatrix}.
$$

Hai họ này không nên bị trộn lẫn trong cùng một thí nghiệm nếu không ghi rõ convention,
vì dấu và pha thay đổi cấu trúc giao thoa dù xác suất một bước có thể trông giống nhau.

Coin tác động đồng nhất tại mọi vị trí qua $C\otimes I_P$:

$$
(C\otimes I_P)|c,x\rangle=(C|c\rangle)\otimes|x\rangle.
$$

## 1.4. Conditional shift

Theo quy ước của dự án, shift operator là

$$
S=
|\uparrow\rangle\langle\uparrow|\otimes
\sum_{x\in\mathbb Z}|x+1\rangle\langle x|
+
|\downarrow\rangle\langle\downarrow|\otimes
\sum_{x\in\mathbb Z}|x-1\rangle\langle x|.
$$

Do đó

$$
S|\uparrow,x\rangle=|\uparrow,x+1\rangle,\qquad
S|\downarrow,x\rangle=|\downarrow,x-1\rangle.
$$

$S$ chỉ hoán vị một cơ sở trực chuẩn nên $S^\dagger S=I$. Trên đường thẳng vô hạn,
không có mất mát amplitude tại biên. Trong mô phỏng hữu hạn phải chọn lattice đủ rộng,
periodic boundary, hoặc reflecting boundary. Phase 1 dùng lattice rộng đúng bằng miền
reachable sau $T$ bước, nên không wrap-around và không cắt amplitude.

## 1.5. Toán tử tiến hóa một bước

Coin được áp dụng trước shift:

$$
U=S(C\otimes I_P),\qquad |\psi_{t+1}\rangle=U|\psi_t\rangle.
$$

Vì tích của các toán tử unitary vẫn unitary,

$$
U^\dagger U
=(C^\dagger\otimes I_P)S^\dagger S(C\otimes I_P)
=(C^\dagger C)\otimes I_P=I.
$$

Đây là invariant quan trọng nhất cho implementation: norm phải bằng một tới sai số số
học. Notebook kiểm tra cả symbolic identity của $H$ và Frobenius norm
$\|U^\dagger U-I\|_F$ trên một cycle hữu hạn.

Với Hadamard coin, recurrence của amplitude theo convention trên là

$$
a_x(t+1)=\frac{a_{x-1}(t)+b_{x-1}(t)}{\sqrt2},
$$

$$
b_x(t+1)=\frac{a_{x+1}(t)-b_{x+1}(t)}{\sqrt2}.
$$

Các recurrence cho thấy nhiều lịch sử coin có thể hội tụ về cùng $(c,x)$ và giao thoa.
Trong CRW, ta cộng xác suất của các lịch sử; trong QRW, ta cộng amplitude phức.

## 1.6. Điều kiện đầu và tính đối xứng

Walker bắt đầu tại gốc:

$$
|\psi_0\rangle=|\chi\rangle\otimes|0\rangle,\qquad
|\chi\rangle=\alpha|\uparrow\rangle+\beta|\downarrow\rangle,
$$

với $|\alpha|^2+|\beta|^2=1$. Hadamard coin không đối xứng theo cách ngây thơ đối với
$|\uparrow\rangle$ hoặc $|\downarrow\rangle$; một trạng thái coin thuần như vậy thường
tạo phân phối lệch. Trạng thái

$$
|\chi_{\rm sym}\rangle
=\frac{|\uparrow\rangle+i|\downarrow\rangle}{\sqrt2}
$$

tạo phân phối vị trí đối xứng với convention đã chọn. Dấu của $i$ có thể đổi khi đổi
quy ước hướng shift hoặc basis; điều cần kiểm tra là $P(x,t)=P(-x,t)$, không phải học
thuộc một dấu pha tách rời convention.

## 1.7. Biểu diễn Fourier và vận tốc nhóm

Do walk đồng nhất theo phép tịnh tiến, Fourier transform tách tiến hóa thành các block
$2\times2$. Với $|k\rangle=\sum_x e^{ikx}|x\rangle$ và convention phù hợp,

$$
S(k)=
\begin{pmatrix}
e^{ik}&0\\
0&e^{-ik}
\end{pmatrix},
\qquad U(k)=S(k)H.
$$

Eigenvalues của mỗi $U(k)$ nằm trên unit circle và có dạng pha
$e^{i\omega_j(k)}$. Đạo hàm $v_g=d\omega/dk$ là group velocity. Với Hadamard walk,
miền vận tốc tiệm cận bị chặn bởi $|v|\le 1/\sqrt2$. Vì vậy hai front nổi bật xuất
hiện gần $x\approx\pm t/\sqrt2$, trong khi walker vẫn bị chặn nhân quả bởi $|x|\le t$.
Stationary-phase analysis của các mode Fourier dẫn đến độ rộng tỷ lệ $t$, tức ballistic
spreading.

## 1.8. Phép đo, density matrix và decoherence

Pure state có density operator

$$
\rho_t=|\psi_t\rangle\langle\psi_t|,
\qquad \rho_{t+1}=U\rho_tU^\dagger.
$$

Một dephasing channel trong basis coin-position có thể viết

$$
\mathcal D_\eta(\rho)
=(1-\eta)\rho
+\eta\sum_{c,x}\Pi_{c,x}\rho\Pi_{c,x},
\quad
\Pi_{c,x}=|c,x\rangle\langle c,x|,
$$

với $0\le\eta\le1$. Map này completely positive và trace preserving (CPTP).
$\eta=0$ giữ coherence; $\eta=1$ xóa toàn bộ off-diagonal terms sau mỗi bước. Nếu
dephase/measure đầy đủ sau từng coin-shift, lịch sử không còn giao thoa và diagonal
probabilities tiến hóa như một Markov chain với transition probabilities
$|C_{ij}|^2$. Với Hadamard coin, giới hạn đó là simple symmetric CRW.

Không phải mọi dạng "noise" đều cho cùng classical limit. Coin-only dephasing,
position-only measurement, broken links và random coin có diffusion coefficient và
thời gian crossover khác nhau. Vì vậy tham số decoherence trong mô hình thị trường phải
đi kèm định nghĩa channel, không chỉ là một số $\gamma$ mơ hồ.

## 1.9. Các invariant dùng để kiểm thử

Implementation Phase 3 nên bảo toàn hoặc kiểm tra:

1. Chuẩn hóa: $\sum_xP(x,t)=1$.
2. Unitary: $\|U^\dagger U-I\|_F$ gần machine precision.
3. Causality: $P(x,t)=0$ nếu $|x|>t$, và parity yêu cầu $x+t$ chẵn.
4. Đối xứng: với initial coin đối xứng, $P(x,t)=P(-x,t)$.
5. Spectrum: $||\lambda_j(U)|-1|$ gần zero trên finite periodic lattice.
6. Pure-state norm: $\|\psi_t\|_2=1$ khi không decoherence và không hấp thụ ở biên.

Các kiểm tra này phát hiện được sai shift direction, off-by-one, boundary leakage và
nhân tensor sai thứ tự trước khi kết quả được diễn giải theo tài chính.

## 1.10. Kết luận

DTQRW một chiều được xác định bởi state space, initial state, coin, shift, boundary và
measurement protocol. Chỉ nói "dùng Hadamard walk" là chưa đủ để tái lập kết quả. Trong
dự án này, $U=S(H\otimes I)$, initial coin là
$(|\uparrow\rangle+i|\downarrow\rangle)/\sqrt2$, và phân phối được đo sau $T$ bước.
Unitary evolution bảo toàn norm; giao thoa giữa các đường đi tạo ballistic spreading;
decoherence có định nghĩa rõ ràng làm suy giảm giao thoa và có thể đưa walk về giới hạn
Markov cổ điển.

## Tài liệu chính

- Aharonov, Davidovich, và Zagury (1993), *Quantum random walks*.
- Kempe (2003), *Quantum random walks: an introductory overview*.
- Konno (2002), *Quantum random walks in one dimension*.
- Kendon (2007), *Decoherence in quantum walks: a review*.
- Venegas-Andraca (2012), *Quantum walks: a comprehensive review*.

