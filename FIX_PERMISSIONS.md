# Fix macOS Permission Issues for AI Trip Planner

## Quick Fix Steps

### Step 1: Grant Terminal Full Disk Access

1. Open **System Settings** (or System Preferences on older macOS)
2. Go to **Privacy & Security** → **Full Disk Access**
3. Click the **lock icon** 🔒 and enter your password
4. Click the **+** button to add an application
5. Navigate to **Applications** → **Utilities** → **Terminal**
6. Select **Terminal** and click **Open**
7. Make sure the checkbox next to Terminal is **checked** ✅
8. Close System Settings

### Step 2: Grant Terminal Network Access (if needed)

1. Open **System Settings** → **Privacy & Security** → **Network**
2. Look for **Terminal** in the list
3. If it's not there or blocked, add it and enable network access

### Step 3: Check Firewall Settings

1. Open **System Settings** → **Network** → **Firewall**
2. If Firewall is ON:
   - Click **Options** or **Firewall Options**
   - Click **+** to add an application
   - Add **Python** (usually located at: `/usr/bin/python3` or your venv Python)
   - Set it to **Allow incoming connections**
   - Click **OK**

### Step 4: Restart Terminal

After making these changes:
1. **Quit Terminal completely** (Cmd+Q)
2. **Reopen Terminal**
3. Try running the server again

## Alternative: Use a Different Port

If permissions still don't work, try using a higher port number (above 1024):

```bash
cd /Users/jenny/Documents/ai-trip-planner
source backend/.venv/bin/activate
cd backend
uvicorn main:app --host 127.0.0.1 --port 3000
```

Then access at: http://localhost:3000

## Verify It's Working

After making changes, test with:

```bash
cd /Users/jenny/Documents/ai-trip-planner
source backend/.venv/bin/activate
cd backend
uvicorn main:app --host 127.0.0.1 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

## Still Having Issues?

If you still get "operation not permitted":
1. Make sure you're using `127.0.0.1` not `0.0.0.0`
2. Try ports: 3000, 5000, 8080, or 8888
3. Check if another app is using port 8000: `lsof -i :8000`
4. Restart your Mac (sometimes helps reset network permissions)
