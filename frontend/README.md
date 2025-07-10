# Synthetic Dataset Generator Frontend

This is the **frontend** for the Synthetic Dataset Generator project, built with [Next.js](https://nextjs.org/) and React.

---

## 🚀 Features
- Modern, responsive UI for uploading datasets and generating synthetic data
- Supports CSV, Excel, TXT, and more
- Domain and privacy level selection
- Download synthetic data results

---

## 🛠️ Getting Started (Local Development)

### 1. **Install Dependencies**
Make sure you have [Node.js](https://nodejs.org/) and [pnpm](https://pnpm.io/) installed.

```bash
cd frontend
pnpm install
```

### 2. **Run the Development Server**
```bash
pnpm dev
```
- The app will be available at [http://localhost:3000](http://localhost:3000)

### 3. **Build for Production**
```bash
pnpm build
pnpm start
```

---

## 🌐 How the Frontend Works
- Built with Next.js (React framework)
- Uses API routes to communicate with the backend (FastAPI)
- Uploads files, selects domain/privacy, and downloads results
- All configuration is in `frontend/`

---

## ☁️ Hot Deploy for Free on AWS EC2 (Ubuntu)

### **1. Launch a Free EC2 Instance**
- Go to [AWS EC2 Console](https://console.aws.amazon.com/ec2/)
- Launch a new instance (choose Ubuntu, t2.micro for free tier)
- Open ports 22 (SSH) and 3000 (for Next.js)

### **2. SSH Into Your Instance**
```bash
ssh -i /path/to/your-key.pem ubuntu@<your-ec2-public-ip>
```

### **3. Install Node.js and pnpm**
```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
npm install -g pnpm
```

### **4. Upload Your Frontend Code**
- You can use `scp`, `rsync`, or push to GitHub and clone:
```bash
git clone https://github.com/<your-username>/synthetic-dataset-generator.git
cd synthetic-dataset-generator/frontend
```

### **5. Install Dependencies and Build**
```bash
pnpm install
pnpm build
```

### **6. Start the App (Hot Deploy)**
```bash
pnpm start
```
- The app will run on port 3000 by default.
- Visit `http://<your-ec2-public-ip>:3000` in your browser.

### **7. (Optional) Keep It Running**
Use [pm2](https://pm2.keymetrics.io/) or `screen`/`tmux` to keep the app running after you disconnect:
```bash
npm install -g pm2
pm2 start "pnpm start"
```

---

## 📝 Notes
- Make sure your backend is also running and accessible to the frontend.
- For production, consider using a reverse proxy (Nginx) and HTTPS.
- Free EC2 instances may sleep or be reclaimed; for persistent hosting, consider AWS Lightsail, Vercel, or Netlify.

---

**Happy coding!** 