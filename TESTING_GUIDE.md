# Testing Guide - Phase 1 Improvements

## 🔁 Regression Coverage – November 2025 Updates

### Test A: Resistivity Temperature & Mud Corrections  
1. Load a LAS file that includes deep resistivity, mud resistivity, and bottom-hole temperature metadata.  
2. Record the raw resistivity curve over a hot interval (>90 °C).  
3. Run processing with environmental corrections enabled.  
4. Confirm the corrected curve decreases relative to the raw data according to the 2 % per 10 °C slope.  
5. Change the mud resistivity input by ±50 % and verify the corrected curve scales using the quarter-power adjustment.  
6. Capture screenshots/log entries that the correction parameters were applied.

### Test B: Depth-Aware Gap Threshold Reporting  
1. Load a dataset with 0.25 m sampling and missing intervals.  
2. Trigger processing and open the processing log.  
3. Validate the reported geological/large gap thresholds list both point counts and metres that correspond (e.g., 400 pts ≈ 100 m).  
4. Repeat with a 1.0 m sampled file and confirm the metre values track the change.

### Test C: RRP Fallback Without SciPy  
1. Temporarily uninstall/disable SciPy (or launch in a virtual environment without it).  
2. Load a file that contains large gaps and run processing.  
3. Confirm the log reports that formation-based gap filling is skipped and the pipeline completes using standard methods.  
4. Re-enable SciPy and confirm formation-based filling is restored.

### Test D: Thread-Safe Error Dialogs  
1. In a processing session, force a controlled failure (e.g., delete the depth curve after loading).  
2. Ensure the error dialog appears, the UI remains responsive, and the log captures the `[ERROR]` entry.  
3. Verify no exceptions are raised in the console about Tkinter thread access.

Document the outcomes in the release checklist before shipping.

## 🧪 HOW TO TEST THE POPUP VISUALIZATION FIX

### Before You Start
The popup visualizations have been completely rewritten to use proper Toplevel windows instead of `plt.show(block=False)`. This should eliminate memory leaks.

---

## ✅ TEST 1: Basic Popup Visualization

### Steps:
1. Load a LAS file
2. Go to Visualization tab
3. Check "Open plots in new window" checkbox
4. Select a curve from dropdown
5. Select visualization type (e.g., "Single Curve")
6. Click "Update Visualization"

### Expected Result:
- ✅ New window opens with matplotlib plot
- ✅ Window title shows viz type and curve name
- ✅ Window title shows well name (e.g., "(Well: ABC-123)")
- ✅ Plot displays correctly with proper axes
- ✅ Navigation toolbar visible at top
- ✅ Window is resizable

### If It Fails:
- Check the Results/Processing Log tab for error messages
- Error should show specific issue (data not found, curve missing, etc.)
- Try unchecking "Open in new window" to use embedded mode

---

## ✅ TEST 2: Multiple Popup Windows

### Steps:
1. With data loaded, create first popup (e.g., "Single Curve" for GR)
2. Leave first window open
3. Select different curve
4. Create second popup (e.g., "Single Curve" for RHOB)
5. Leave both open
6. Create third popup (different visualization type)

### Expected Result:
- ✅ All popup windows remain open simultaneously
- ✅ Each window is independent and functional
- ✅ Can interact with all windows
- ✅ No performance degradation
- ✅ Memory usage reasonable (check Task Manager)

---

## ✅ TEST 3: Popup Cleanup on Close

### Steps:
1. Create 2-3 popup visualizations
2. Close them one by one using the X button
3. Check Task Manager memory usage
4. Check Results log for cleanup messages

### Expected Result:
- ✅ Each window closes cleanly
- ✅ Log shows "Closed popup visualization: [type]"
- ✅ Memory is freed (check Task Manager - Python process should drop)
- ✅ No error messages

---

## ✅ TEST 4: Popup Cleanup on New Well Load

### Steps:
1. Load a well (Well A)
2. Create 2-3 popup visualizations
3. Load a different well (Well B) - or reload same file
4. Check if popup windows close automatically
5. Check Results log

### Expected Result:
- ✅ All popup windows close automatically when loading new well
- ✅ Log shows "Closed N popup visualization windows"
- ✅ Memory freed
- ✅ No popup windows from Well A remain when Well B loads

---

## ✅ TEST 5: Well Identification in Popups

### Steps:
1. Load a well with well name in LAS header
2. Create popup visualization
3. Check window title

### Expected Result:
- ✅ Window title includes well name: "Single Curve - GR (Well: ABC-123)"
- ✅ Well name matches what's shown in main window title
- ✅ Well name matches Data Tab well info card

---

## ✅ TEST 6: Data Source Handling

### Test 6A: Before Processing
1. Load file (don't process yet)
2. Create popup visualization
3. Should show original data in red

### Test 6B: After Processing
1. Load file
2. Process data
3. Create popup visualization
4. Should show processed data in blue/green

### Expected Result:
- ✅ Popups work both before and after processing
- ✅ Correct data source selected automatically
- ✅ Legend shows "Original" or "Processed" correctly

---

## ✅ TEST 7: Error Handling

### Steps:
1. Try creating popup with no data loaded
2. Try creating popup with invalid curve selected
3. Close popup during plot creation

### Expected Result:
- ✅ Helpful error message appears
- ✅ Error explains what's wrong (no data, curve not found, etc.)
- ✅ Suggests solution (load data, select valid curve)
- ✅ App doesn't crash

---

## ✅ TEST 8: Memory Leak Verification

### Steps:
1. Note Python process memory in Task Manager (baseline)
2. Create 5 popup visualizations
3. Note memory increase
4. Close all 5 popups
5. Wait 5 seconds
6. Note memory after closure

### Expected Result:
- ✅ Memory increases when popups open (normal)
- ✅ Memory decreases significantly after closing popups
- ✅ Memory returns close to baseline (within 10-20MB)
- ✅ No continuous memory growth

---

## 🐛 IF POPUPS DON'T WORK - TROUBLESHOOTING

### Issue: Window opens but plot doesn't show
**Check:**
- Is data loaded? (Check Data Tab)
- Is curve selected? (Check Visualization tab dropdown)
- Any errors in Results log?

### Issue: Error message appears
**Check:**
- What does the error say specifically?
- Check Results/Processing Log for detailed traceback
- Verify data is loaded and processed

### Issue: Window is blank/empty
**Check:**
- Does embedded mode work? (uncheck "Open in new window")
- Is the plotting method implemented for that viz type?
- Check if data exists for selected curve

### Issue: Memory doesn't decrease after closing
**Check:**
- Wait 10-15 seconds (garbage collection has delay)
- Try loading a new well (triggers cleanup)
- Check Results log for cleanup messages

---

## 📊 PERFORMANCE EXPECTATIONS

### Normal Behavior:
- Popup window opens in < 1 second
- Plot renders immediately
- Window is responsive
- Close is instant
- Memory freed within 10-15 seconds

### Warning Signs:
- ⚠️ Popups take > 3 seconds to open (check data size)
- ⚠️ Memory doesn't decrease after closing (check for errors)
- ⚠️ Windows freeze or lag (check processing log)

---

## 💡 WHAT'S DIFFERENT NOW

### Before (plt.show):
- ❌ Memory leaks
- ❌ Event loop conflicts
- ❌ No cleanup tracking
- ❌ Figures accumulate in memory

### After (Toplevel windows):
- ✅ Proper memory management
- ✅ No event loop conflicts
- ✅ Cleanup tracking via registry
- ✅ Figures freed on close

---

## 🎯 SUCCESS CRITERIA

If ALL these work, the fix is successful:
- ✅ Popups open and display correctly
- ✅ Multiple popups work simultaneously
- ✅ Closing popups frees memory
- ✅ Loading new well closes all popups
- ✅ Well name appears in popup titles
- ✅ No error messages
- ✅ No memory accumulation

---

**If popups are still not working, please:**
1. Check the Results/Processing Log for specific error messages
2. Try one simple popup first (Single Curve visualization)
3. Let me know what error appears - I'll fix it immediately

**The software should feel "lighter" because:**
- Memory is now properly freed when popups close
- No accumulation of hidden matplotlib figures
- Garbage collection working correctly

