# ComfyUI-SpeechBubble

이미지에 다양하고 예쁜 말풍선을 손쉽게 추가할 수 있는 ComfyUI 커스텀 노드입니다. 직접 설정한 기하학적 도형으로 말풍선을 생성하거나, 사용자가 가지고 있는 커스텀 말풍선 이미지를 입력받아 텍스트를 얹어 합성할 수 있습니다.

---

## 🌟 주요 기능 (Features)

| 기능 | 설명 |
|------|------|
| **도형 생성형 말풍선** | `ellipse`(타원) / `rounded_rectangle`(둥근 사각형) / `rectangle`(직사각형) 지원 |
| **커스텀 이미지 말풍선** | `bubble_image` 입력을 통해 외부 말풍선 이미지(RGBA/RGB)를 배치하고 텍스트 자동 래핑 합성 |
| **위치 조절 가능한 꼬리** | `tail_x`, `tail_y` 좌표로 말하는 대상의 입 위치 등에 맞추어 꼬리 끝 조정 가능 |
| **테두리 제거 기능** | `border_width`를 `0`으로 설정하여 테두리 없는 깔끔한 말풍선 표현 가능 |
| **손그림(HandDrawn) 효과** | 외곽선을 손으로 삐뚤삐뚤하게 그린 듯한 효과를 주는 노이즈 옵션 지원 |
| **이중 출력 지원** | 합성된 이미지(RGB) + 말풍선 레이어 단독(RGBA) + 알파 채널 마스크(MASK) 분리 출력 |
| **레이아웃 저장/불러오기** | 말풍선 위치, 크기, 꼬리 좌표, 텍스트 설정을 JSON 문자열로 직렬화하여 재사용 가능 |
| **한글 및 다양한 폰트 지원** | TTF / OTF / TTC 폰트 파일을 폴더에 넣어주면 드롭다운 메뉴에 자동 등록 |

---

## ⚙️ 설치 방법 (Installation)

1. ComfyUI의 custom_nodes 폴더 아래로 이 저장소를 클론하거나 압축을 해제합니다.
   ```bash
   cd ComfyUI/custom_nodes/
   git clone https://github.com/Toraong/ComfyUI-SpeechBubble.git
   ```

2. 해당 폴더 내의 필요한 파이썬 패키지를 설치합니다. (Pillow, numpy, torch 필요)
   ```bash
   pip install -r requirements.txt
   ```

3. **폰트 설치 (선택사항)**: 사용하려는 한글 및 영문 폰트 파일(`.ttf`, `.otf`, `.ttc`)을 아래 경로에 넣어주시면 노드의 `font_name` 목록에 자동으로 추가됩니다.
   ```
   ComfyUI-SpeechBubble/
   └── assets/
       └── font/       <-- 여기에 폰트 파일을 복사하세요.
   ```

---

## 🧩 노드 정보 (Node Details)

### 1. 🗨️ Speech Bubble (SpeechBubble)
가장 핵심이 되는 메인 합성 노드입니다.

#### 입력 파라미터 (Inputs)
- **`image`** (IMAGE): 말풍선을 합성할 배경 이미지
- **`text`** (STRING, 멀티라인 지원): 말풍선 내부에 들어갈 텍스트 내용
- **`bubble_x` / `bubble_y`** (INT): 말풍선 바디의 좌상단 시작 좌표 (X, Y)
- **`bubble_w` / `bubble_h`** (INT): 말풍선 바디의 너비와 높이
- **`tail_x` / `tail_y`** (INT): 말풍선 꼬리 끝이 가리킬 절대 좌표 (도형 말풍선 사용 시 활성화)
- **`shape`** (선택): `ellipse` (타원) / `rounded_rectangle` (둥근 사각형) / `rectangle` (직사각형)
- **`fill_color`** (STRING): 배경 채우기 색상 (Hex 코드 입력, 예: `#FFFFFF`)
- **`border_color`** (STRING): 테두리 외곽선 색상 (Hex 코드 입력, 예: `#000000`)
- **`text_color`** (STRING): 텍스트 글자 색상 (Hex 코드 입력, 예: `#000000`)
- **`border_width`** (INT): 테두리 두께. `0`으로 설정하면 테두리가 나타나지 않습니다.
- **`opacity`** (INT, 0 ~ 255): 말풍선 자체의 불투명도 (255: 완전히 불투명)
- **`font_name`** (선택): 사용할 폰트명 (`assets/font`에 있는 폰트가 자동 검출됨)
- **`font_size`** (INT): 텍스트 글자 크기
- **`padding`** (INT): 말풍선 테두리와 텍스트 간의 내부 여백
- **`line_spacing`** (INT): 텍스트 줄바꿈 시 줄 간 여백
- **`tail_width`** (FLOAT, 0.01 ~ 0.5): 말풍선 크기 대비 꼬리가 연결되는 부분의 너비 비율
- **`handdrawn`** (BOOLEAN): 활성화 시 외곽선에 미세한 굴곡을 주어 손으로 그린 듯한 느낌을 줍니다.
- **`handdrawn_strength`** (FLOAT): 손그림 효과의 강도
- **`seed`** (INT): 손그림 노이즈 굴곡을 결정하는 난수 시드 값
- **`bubble_image`** (IMAGE, *선택*): **[New]** 사용자가 준비한 말풍선 이미지를 입력합니다. 연결 시 도형 생성 로직 대신 입력된 이미지가 지정한 `bubble_x/y/w/h` 위치와 크기로 자동 리사이징 및 배치되며, 내부에 텍스트가 배치됩니다.
- **`settings_json`** (STRING, *선택*): `SpeechBubbleSettings` 노드에서 저장한 레이아웃 JSON을 연결하여 지오메트리 옵션을 한 번에 오버라이드할 수 있습니다.

#### 출력 (Outputs)
- **`composited`** (IMAGE): 말풍선이 배경 이미지 위에 자연스럽게 합성된 최종 이미지 (RGB)
- **`bubble_rgba`** (IMAGE): 투명 배경 위에 말풍선 레이어만 얹어진 레이어 이미지 (RGBA)
- **`bubble_mask`** (MASK): 말풍선 영역만을 분리해 낸 알파 마스크 채널 (0.0 ~ 1.0)

---

### 2. 🗨️ Speech Bubble Settings (SpeechBubbleSettings)
말풍선의 위치(`bubble_x`, `bubble_y`, `bubble_w`, `bubble_h`), 꼬리 끝 좌표(`tail_x`, `tail_y`), 그리고 텍스트 내용을 한 개의 **JSON String**으로 묶어주는 유틸리티 노드입니다. 
이 값을 `SpeechBubble` 노드의 `settings_json`에 전달하면, 워크플로우 상에서 개별 설정을 따로 보관하거나 재사용하기에 매우 간편합니다.
