import cv2
import numpy as np
from typing import List


def combine_4_videos_into_one_elad(
        video_paths: List[str],
        output_path: str,
        show_live: bool = False
) -> None:
    """
    מאחד ארבעה סרטונים לסרטון אחד (2x2).

    Args:
        video_paths (List[str]): רשימת נתיבי קבצי הווידאו (חייבת להיות בגודל 4).
        output_path (str): נתיב לשמירת הסרטון המאוחד.
        show_live (bool): האם להציג את הסרטון בלייב תוך כדי הרצה.
    """
    if len(video_paths) != 4:
        raise ValueError("יש לספק בדיוק ארבעה נתיבי סרטונים.")

    # פתיחת קבצי הווידאו
    caps = [cv2.VideoCapture(path) for path in video_paths]
    if not all(cap.isOpened() for cap in caps):
        raise IOError("אחד או יותר מהסרטונים לא נפתחו בהצלחה.")

    # בדיקת מספר פריימים
    frame_counts = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
    if len(set(frame_counts)) != 1:
        raise ValueError("לסרטונים יש מספר פריימים שונה.")

    # הגדרת פרמטרים בסיסיים
    fps = caps[0].get(cv2.CAP_PROP_FPS)
    frame_width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    combined_size = (frame_width * 2, frame_height * 2)

    # הגדרת VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, combined_size)
    if not out.isOpened():
        raise IOError("נכשל ביצירת קובץ הווידאו המאוחד.")

    print("מתחילים עיבוד פריימים...")

    while True:
        # קריאת פריים מכל סרטון
        rets_frames = [cap.read() for cap in caps]
        rets, frames = zip(*rets_frames)

        if not all(rets):
            print("הגעת לסוף אחד הסרטונים.")
            break

        # שינוי גודל במידת הצורך
        resized_frames = [
            cv2.resize(frame, (frame_width, frame_height))
            if (frame.shape[1] != frame_width or frame.shape[0] != frame_height)
            else frame
            for frame in frames
        ]

        # יצירת השורות והמסגרת המשולבת
        top_row = np.hstack(resized_frames[:2])
        bottom_row = np.hstack(resized_frames[2:])
        combined_frame = np.vstack((top_row, bottom_row))

        out.write(combined_frame)

        if show_live:
            cv2.imshow('Combined Video', combined_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("הפסקת תהליך לפי בקשת המשתמש.")
                break

    # ניקוי משאבים
    for cap in caps:
        cap.release()
    out.release()
    if show_live:
        cv2.destroyAllWindows()

    print(f"סיום בהצלחה. קובץ הפלט נשמר ב- {output_path}")


if __name__ == '__main__':
    video_paths = [
        'C:\\Users\\elad2\\Downloads\\vidvid.mp4',
        'C:\\Users\\elad2\\Downloads\\vidvid.mp4',
        'C:\\Users\\elad2\\Downloads\\vidvid.mp4',
        'C:\\Users\\elad2\\Downloads\\vidvid.mp4'
    ]
    output_filename = 'C:\\Users\\elad2\\Downloads\\combined_video_2.mp4'
    combine_4_videos_into_one_elad(video_paths, output_filename, show_live=True)