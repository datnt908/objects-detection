# Cat and Dog Detection with TensorFlow	

Đây là Project cuối kỳ bộ môn __Một số thuật toán thông minh__.  
Thuộc giáo trình Khoa __Công Nghệ Phần Mềm__ của __ĐẠI HỌC CÔNG NGHỆ THÔNG TIN - ĐHQG-TPHCM__.

## Giới thiệu

Đây là repository chứa source code của thư mục workspace cho đồ án. Các dữ liệu và tập tin có dung lượng lớn sẽ được chứa trên các dịch vụ lưu trữ Cloud. Cách sử dụng sẽ được hướng dẫn trong các phần bên dưới.


## Tổng quan về Object Detection

Computer Vision là một lĩnh vực của AI bao gồm thu nhận, xử lý, phân tích, nhận dạng hình ảnh và video để đạt được giá trị thông tin của hình ảnh và video đó.

Computer Vision bao gồm các nhiệm vụ:

* Classification: phân loại một bức ảnh hoạt video
* Localization: Xác định vị trí của đối tượng chính có trong ảnh hoặc video
* __Object Detection__: Tìm kiếm các đối tượng có trong ảnh, xác định vị trí (Localization) cũng như phân loại (Classification) đối tượng đó
* Instance Segmention: Tương tự như Object Detection nhưng (boundary box) chính xác hơn


## Deep Learning cho Object Detection

Project này sử dụng Deep Learning với thuật toán Single Shot Detector (SSD).

SSD sử dụng kiến trúc của mạng VGG16 gồm 5 khối lớp tích chập. Sau khi tích chập và rút trích đặt trưng ở lớp tính chập Conv4_3 (Lớp thứ 3 trong khối tích chập thứ 4) sẽ được Feature Map. Nó được dùng để tiến hành detect object.

Default Boundary Box (Anchor) là các boundary box được đặt sẵn kích thước phù hợp với hình dạng mà đối tượng cần detect. Nó sẽ là cơ sở để dự đoán ra boundary box chính xác hơn.

Multi-scale Feature maps là các Feature Map với kích thước khác nhau nhằm thu được các đặc trưng ở những mức độ khác nhau. Bởi vì có thể các đối tượng cần detect tuy cùng một loại nhưng lại có kích thước khác nhau.

## Chuẩn bị dữ liệu

Dữ liệu được thu thập là bộ hình ảnh về chó và mèo: Mèo nhà lông dài, Mèo Ba Tư, Chó Pug, Chó Husky Siberia.

Tiền xử lý dữ liệu: Đánh chú thích (annotate) cho dữ liệu hình ảnh bằng ứng dụng __LabelImg__. Việc đánh chú thích này là chỉ ra bounding box và loại của các đối tượng (mèo nào, chó nào) cần detect có trong ảnh, và tạo tập tin lưu giữ thông tin.

Chi bộ dữ liệu đã chú thích theo tỉ lệ 9:1, trong đó 9 phần dùng để training, phần còn lại để đánh giá và kiểm tra.

Cuối cùng tạo các định dạng dữ liệu phù hợp với model chúng ta lựa chọn. Ở đây là TensorRecord vì triển khai trên TensorFlow.


## Mô hình Nhận diện chó và mèo
ác
SSD ngoài phần VGG16 ra sẽ bổ sung thêm 6 lớp tích chập tạo thành các Multi-scale Layer. Trong đó 5 lớp sẽ trực tiếp tham gia dự đoán object, 3 trong số đó sẽ dự đoán ra 6 trường hợp, còn lại thì dự đoán 4 trường hợp.

Sau khi qua hết tất cả các lớp tích chập sẽ thu được các dự đoán và được gôm vào trong Detections. Như vậy với kiểu kiến trúc của SSD thì ta có được tổng cộng là 8732 dự đoán cùng với score để i.

Tổng hợp kết quả: Vì SSD có quá nhiều kết quả được dự đoán (8732) nên cần phải loại bỏ dự đoán thừa - dự đoán có độ tin cập thấp hơn 0.01. Ngoài ra, chúng ta sử dụng __non-maximum-suppression__ để loại bỏ các dự đoán trùng lắp (cùng dự đoán 1 object).

## Đánh giá độ chính xác

Để đánh giá một hình ảnh được Object Detection model dự đoán ra, chúng ta sẽ có danh sách các Ground Truth và Prediction Box của hình ảnh đó. Ground Truth là boundary box do người đánh chú thích - được dùng để kiểm chứng. Prediction Box là boundary do model sau khi train sẽ dự đóan ra.

Với mỗi Prediction Box nào có giá trị IoU (Intersection over Union) với Ground Truth tương ứng mà lớn hơn 0.5 thì được xem là khớp (matched), ngược lại là không khớp. Độ chính xác của việc detect object trên 1 hình ảnh được tính dựa trên số lượng Prediction Box khớp.