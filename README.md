<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>即将上线</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    .blink {
      animation: blink 1.5s infinite;
    }

    @keyframes blink {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.4; }
    }
  </style>
</head>
<body class="bg-white text-center flex items-center justify-center h-screen flex-col">
  <h1 class="text-5xl font-bold text-gray-800">🚧 敬请期待 🚧</h1>
  <p class="text-xl mt-4 text-gray-600">我们正在努力开发中，敬请期待！</p>
  <p class="mt-6 text-lg text-blue-500 blink">Coming Soon...</p>
</body>
</html>
