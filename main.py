import cv2
import datetime
import easyocr
import mysql.connector

connection = mysql.connector.connect(host='localhost',user='root',password='',database='license_plate_db')
db_cursor = connection.cursor()

harcascade = "model/haarcascade_russian_plate_number.xml"

cap = cv2.VideoCapture(0)

cap.set(3, 640)  # width
cap.set(4, 480)  # height

# Define the coordinates of the region of interest (ROI)
roi_x, roi_y, roi_width, roi_height = 200, 100, 400, 300

min_area = 500
max_total_area = 20000  # Maximum total area of detected plates
max_plate_count = 2  # Maximum number of plates to detect
count = 0
total_area = 0
img_count = 0

reader = easyocr.Reader(['en'])

while True:
    success, img = cap.read()

    # Crop the frame to focus on the ROI
    roi = img[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date, time = current_time.split()  # Splitting the current time into date and time
    cv2.putText(img, date, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 2)  # Displaying date
    cv2.putText(img, time, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 2)  # Displaying time

    plate_cascade = cv2.CascadeClassifier(harcascade)
    img_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    # Draw the green rectangle for the ROI
    cv2.rectangle(img, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (0, 255, 0), 2)

    for (x, y, w, h) in plates:
        area = w * h

        if area > min_area:
            # Convert coordinates to the original frame
            x += roi_x
            y += roi_y
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            img_roi = img[y: y + h, x:x + w]

            # Threshold the image
            _, img_thresholded = cv2.threshold(cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY), 0, 255,
                                               cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            
            result = reader.readtext(img_roi)

            if result:
                plate_text = result[0][1]
                if len(plate_text) == 8 and plate_text[:3].isalpha() and plate_text[3] == ' ' and plate_text[
                                                                                                  4:].isdigit():  # Check if plate text has 3 letters, a space, and 4 numbers
                    plate_text_size = cv2.getTextSize(plate_text, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 2)[0]
                    plate_text_x = x + (w - plate_text_size[0]) // 2
                    plate_text_y = y - 10  # Adjusted to place text at the top of the bounding box
                    cv2.putText(img, plate_text, (plate_text_x, plate_text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                                (255, 0, 255), 2)

                    print("License Plate:", plate_text, "Detected at:", current_time)

                    # check license plate if registered in database and insert the data to whether its registered or not registered
                    check_plate = "SELECT * FROM registered WHERE license_plate = %s"
                    db_cursor.execute(check_plate, (plate_text,))
                    result = db_cursor.fetchone()
                    if result:
                        insert_db = "INSERT INTO monitoring(license_plate, date, time, registered) VALUES (%s,%s,%s,%s)"
                        insert_values = (plate_text, date, time, "registered")
                        db_cursor.execute(insert_db, insert_values)
                        connection.commit()
                        print("License plate already registered".format(plate_text))
                    else:
                        insert_db = "INSERT INTO monitoring(license_plate, date, time, registered) VALUES (%s,%s,%s,%s)"
                        insert_values = (plate_text, date, time, "Not registered")
                        db_cursor.execute(insert_db, insert_values)
                        connection.commit()
                        print("Not registered")

           
            if count < max_plate_count:
                cv2.imwrite("plates/scanned_" + str(img_count) + ".jpg", img_thresholded)
                count += 1
                img_count += 1

    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


db_cursor.close()
connection.close()

cap.release()
cv2.destroyAllWindows()
