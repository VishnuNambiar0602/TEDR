# TEDR UI/UX Documentation

This document describes the user interface and user experience of the TEDR application.

## Visual Design

### Color Palette

**Primary Colors:**
- Primary Blue: `#3B82F6` - Used for vehicles, primary buttons
- Secondary Purple: `#8B5CF6` - Used for gradients, accents
- Success Green: `#10B981` - Used for pedestrians, success states
- Warning Orange: `#F97316` - Used for animals
- Info Yellow: `#FACC15` - Used for traffic elements

**Background:**
- Gradient: `linear-gradient(135deg, #667eea 0%, #764ba2 100%)`
- Card Background: `rgba(30, 41, 59, 0.8)` with backdrop blur
- Dark Background: `#0F172A`

**Text:**
- Light Text: `#F8FAFC`
- Muted Text: `#94A3B8`

### Typography

**Font Family:** Poppins (Google Fonts)
- Headings: 600-700 weight
- Body: 400 weight
- Light elements: 300 weight

**Font Sizes:**
- Logo: 2.5rem (40px)
- Section Title: 1.8rem (29px)
- Card Title: 1.3rem (21px)
- Body: 1rem (16px)
- Small: 0.9rem (14px)

### Layout

**Container:**
- Max width: 1200px
- Responsive padding: 20px

**Grid:**
- Statistics cards: CSS Grid with `auto-fit` and `minmax(200px, 1fr)`
- Responsive breakpoints:
  - Desktop: 768px+
  - Tablet: 480px - 768px
  - Mobile: < 480px

## User Interface Components

### 1. Header Section

```
┌──────────────────────────────────────────────────────┐
│  🚗 TEDR                                             │
│  AI-Powered Object Detection for Indian Roads       │
└──────────────────────────────────────────────────────┘
```

**Features:**
- Dark background with blur effect
- Large logo with car icon
- Centered tagline

### 2. Upload Section (Initial State)

```
┌────────────────────────────────────────────────────────┐
│  📤 Upload Image                                       │
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │                                                  │ │
│  │                   🖼️                              │ │
│  │                                                  │ │
│  │     Drag & Drop your image here                 │ │
│  │           or click to browse                     │ │
│  │                                                  │ │
│  │   Supported: JPG, PNG, JPEG, WebP (Max 10MB)    │ │
│  │                                                  │ │
│  └──────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

**Features:**
- Dashed border that changes on hover/drag
- Large icon (64px)
- Centered text with clear instructions
- Hover effect: border turns blue, slight scale up
- Drag-over effect: border turns green, scale up more

### 3. Upload Section (With Preview)

```
┌────────────────────────────────────────────────────────┐
│  📤 Upload Image                                       │
│                                                        │
│  Preview                                              │
│  ┌──────────────────────────────────────────────────┐ │
│  │                                                  │ │
│  │         [Image Preview]              ❌          │ │
│  │                                                  │ │
│  └──────────────────────────────────────────────────┘ │
│                                                        │
│            [ 🔍 Detect Objects ]                      │
└────────────────────────────────────────────────────────┘
```

**Features:**
- Image preview with max height 400px
- Remove button (X) in top-right corner
- Primary action button: "Detect Objects"
- Button has gradient background
- Button hover: lifts up with shadow

### 4. Loading Section

```
┌────────────────────────────────────────────────────────┐
│                                                        │
│                       ⭕                                │
│                   (spinning)                           │
│                                                        │
│         AI is analyzing your image...                 │
│       Please wait while we detect objects             │
│                                                        │
└────────────────────────────────────────────────────────┘
```

**Features:**
- Centered spinner animation
- Smooth rotation animation
- Clear status message
- Card disappears after detection completes

### 5. Results Section

```
┌────────────────────────────────────────────────────────┐
│  ✓ Detection Results                                   │
│                                                        │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐     │
│  │    5    │ │    3    │ │    1    │ │    0    │     │
│  │ 🚗      │ │ 🚶      │ │ 🐾      │ │ 🚦      │     │
│  │Vehicles │ │Pedestri-│ │ Animals │ │ Traffic │     │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘     │
│  ┌─────────┐                                          │
│  │    2    │                                          │
│  │ 📦      │                                          │
│  │ Others  │                                          │
│  └─────────┘                                          │
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │                                                  │ │
│  │         [Annotated Image with Boxes]            │ │
│  │                                                  │ │
│  └──────────────────────────────────────────────────┘ │
│                                                        │
│     [ ⬇️ Download Result ] [ 📤 Upload Another ]     │
│                                                        │
│  Detected Objects                                     │
│  ┌──────────────────────────────────────────────────┐ │
│  │ 1. car (vehicle)                        95.3%   │ │
│  ├──────────────────────────────────────────────────┤ │
│  │ 2. person (pedestrian)                  92.1%   │ │
│  ├──────────────────────────────────────────────────┤ │
│  │ 3. motorcycle (vehicle)                 88.7%   │ │
│  └──────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────┘
```

**Features:**
- Statistics cards with colored left border
- Large numbers showing counts
- Icons for each category
- Annotated image display
- Two action buttons (Download, Upload Another)
- Detailed detection list with confidence percentages
- Smooth fade-in animations

### 6. Bounding Boxes on Images

**Box Styling:**
```
Vehicle (Blue):
┌─────────────────┐
│ car: 95.3%      │  ← Blue background label
└─────────────────┘
│                 │
│    [Vehicle]    │  ← Blue border (3px solid)
│                 │
└─────────────────┘

Pedestrian (Green):
┌─────────────────┐
│ person: 92.1%   │  ← Green background label
└─────────────────┘
│                 │
│    [Person]     │  ← Green border (3px solid)
│                 │
└─────────────────┘
```

**Color Mapping:**
- 🔵 Vehicles: Blue (#3B82F6)
- 🟢 Pedestrians: Green (#10B981)
- 🟠 Animals: Orange (#F97316)
- 🟡 Traffic: Yellow (#FACC15)
- 🟣 Others: Purple (#A855F7)

### 7. Footer Section

```
┌──────────────────────────────────────────────────────┐
│  © 2024 TEDR - Transformer-based Object Detection   │
│              for Indian Roads                        │
│                                                      │
│      🔗 GitHub    📖 Documentation                  │
└──────────────────────────────────────────────────────┘
```

**Features:**
- Dark background
- Centered text
- Social links with icons
- Links change color on hover

### 8. Toast Notifications

```
┌──────────────────────────────────┐
│ ℹ️  Image loaded successfully!   │
└──────────────────────────────────┘
```

**Variants:**
- **Success**: Green left border, checkmark icon
- **Error**: Red left border, error icon
- **Info**: Blue left border, info icon

**Behavior:**
- Slides in from right
- Auto-dismisses after 3 seconds
- Position: Fixed, bottom-right

## User Flows

### Flow 1: Successful Detection

1. **User lands on page**
   - Sees header and upload area
   - Gradient background loads

2. **User uploads image**
   - Drags image onto upload area
   - OR clicks to browse files
   - File is validated (type, size)
   - Toast: "Image loaded successfully!"

3. **Image preview appears**
   - Preview section fades in
   - Image displayed with remove button
   - "Detect Objects" button available

4. **User clicks "Detect Objects"**
   - Upload section hides
   - Loading section appears with spinner
   - Message: "AI is analyzing your image..."

5. **Detection completes**
   - Loading section hides
   - Results section fades in
   - Statistics cards populate
   - Annotated image displays
   - Detection list appears

6. **User reviews results**
   - Sees bounding boxes on image
   - Reviews statistics by category
   - Checks detailed detection list

7. **User downloads result (optional)**
   - Clicks "Download Result"
   - Image saves with timestamp
   - Toast: "Image downloaded successfully!"

8. **User uploads another (optional)**
   - Clicks "Upload Another"
   - Returns to initial state
   - Can upload new image

### Flow 2: Error Handling

**Invalid File Type:**
1. User uploads .pdf file
2. Toast appears: "Invalid file type. Please upload JPG, PNG, or WebP image."
3. Upload area remains, no preview

**File Too Large:**
1. User uploads 15MB image
2. Toast appears: "File too large. Maximum size is 10MB."
3. Upload area remains, no preview

**Server Error:**
1. Detection fails on backend
2. Loading section hides
3. Upload section reappears
4. Toast appears: "An error occurred during detection."

## Responsive Design

### Desktop (>768px)
- Full layout with statistics grid (4-5 columns)
- Large images and buttons
- Sidebar-ready layout

### Tablet (480px - 768px)
- Statistics grid: 2-3 columns
- Medium-sized images
- Stacked action buttons

### Mobile (<480px)
- Statistics grid: 1 column
- Full-width buttons
- Compact header
- Smaller padding and margins

## Accessibility Features

- ✅ Semantic HTML (header, main, footer, section)
- ✅ ARIA labels where needed
- ✅ Keyboard navigation support
- ✅ Clear focus indicators
- ✅ High contrast text
- ✅ Descriptive alt text for images
- ✅ Clear error messages

## Animation Details

**Page Load:**
- Elements fade in (0.5s ease-in)
- Slight upward translation

**Upload Area Hover:**
- Border color transition (0.3s)
- Scale transform (1.02x)

**Button Hover:**
- Lift effect (translateY -2px)
- Shadow enhancement
- 0.3s transition

**Spinner:**
- Continuous rotation (1s linear infinite)

**Results Appear:**
- Fade in (0.5s)
- Stats cards stagger slightly

**Toast:**
- Slide in from right (0.3s)
- Fade out (0.3s) before dismiss

## Browser Compatibility

**Supported:**
- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

**Features Used:**
- CSS Grid
- CSS Custom Properties
- Flexbox
- ES6 JavaScript (async/await, fetch)
- FileReader API
- FormData API

## Performance Optimizations

**CSS:**
- Single stylesheet (no imports)
- Minimal use of shadows
- Hardware-accelerated transforms
- Efficient selectors

**JavaScript:**
- Event delegation where possible
- Debounced resize handlers
- Lazy image loading
- Efficient DOM updates

**Images:**
- Base64 for results (no extra requests)
- Responsive image sizing
- Proper image compression

## Future UI Enhancements

Ideas for v2.0:
- [ ] Dark/Light theme toggle
- [ ] Image comparison slider (before/after)
- [ ] Zoom functionality for results
- [ ] Batch upload with gallery view
- [ ] Advanced settings panel
- [ ] Keyboard shortcuts
- [ ] Undo/Redo functionality
- [ ] Export results as PDF
- [ ] Share functionality
- [ ] Multi-language support

---

**Design Philosophy:**
The UI is designed to be:
- **Simple**: Clear, uncluttered interface
- **Modern**: Gradients, glassmorphism, smooth animations
- **Intuitive**: Natural drag-and-drop, clear CTAs
- **Responsive**: Works on all devices
- **Accessible**: Usable by everyone
- **Fast**: Optimized for performance
