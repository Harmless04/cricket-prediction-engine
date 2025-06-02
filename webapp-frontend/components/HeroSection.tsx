'use client'

import { motion } from 'framer-motion'
import { useState, useEffect } from 'react'

const CricketBall = () => (
  <motion.div
    className="w-6 h-6 bg-red-500 rounded-full shadow-lg"
    animate={{
      x: [0, 100, 200, 300],
      y: [0, -50, -20, 0],
      rotate: [0, 180, 360, 540],
    }}
    transition={{
      duration: 2,
      repeat: Infinity,
      repeatDelay: 1,
      ease: "easeInOut"
    }}
  />
)

const StatsCounter = ({ end, label, duration = 2 }: { end: number, label: string, duration?: number }) => {
  const [count, setCount] = useState(0)

  useEffect(() => {
    let startTime: number
    const animate = (currentTime: number) => {
      if (!startTime) startTime = currentTime
      const progress = Math.min((currentTime - startTime) / (duration * 1000), 1)
      setCount(Math.floor(progress * end))
      if (progress < 1) {
        requestAnimationFrame(animate)
      }
    }
    requestAnimationFrame(animate)
  }, [end, duration])

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.5 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.6, delay: 1.2 }}
      className="text-center"
    >
      <motion.div 
        className="text-3xl md:text-4xl font-bold text-white mb-2"
        animate={{ scale: [1, 1.1, 1] }}
        transition={{ duration: 2, repeat: Infinity, repeatDelay: 3 }}
      >
        {count}+
      </motion.div>
      <div className="text-gray-300 text-sm">{label}</div>
    </motion.div>
  )
}

export default function HeroSection() {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Animated Cricket Field Background */}
      <div className="absolute inset-0">
        <motion.div
          className="absolute inset-0 bg-gradient-to-r from-green-400/20 to-emerald-600/20"
          animate={{ 
            backgroundPosition: ['0% 50%', '100% 50%', '0% 50%'],
          }}
          transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
        />
        
        {/* Cricket Field Lines */}
        <svg className="absolute inset-0 w-full h-full opacity-10" viewBox="0 0 800 600">
          <motion.circle
            cx="400"
            cy="300"
            r="100"
            fill="none"
            stroke="white"
            strokeWidth="2"
            initial={{ strokeDasharray: "0 628" }}
            animate={{ strokeDasharray: "628 628" }}
            transition={{ duration: 2, delay: 0.5 }}
          />
          <motion.circle
            cx="400"
            cy="300"
            r="200"
            fill="none"
            stroke="white"
            strokeWidth="1"
            initial={{ strokeDasharray: "0 1256" }}
            animate={{ strokeDasharray: "1256 1256" }}
            transition={{ duration: 3, delay: 1 }}
          />
        </svg>
      </div>

      <div className="relative z-10 text-center px-4 max-w-6xl mx-auto">
        {/* Main Title */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="mb-8"
        >
          <motion.h1 
            className="text-6xl md:text-8xl font-bold bg-gradient-to-r from-white via-blue-200 to-purple-200 bg-clip-text text-transparent mb-6"
            animate={{ 
              backgroundPosition: ['0% 50%', '100% 50%', '0% 50%'],
            }}
            transition={{ duration: 5, repeat: Infinity, ease: "linear" }}
          >
            CricPredict
            <motion.span 
              className="block text-5xl md:text-7xl bg-gradient-to-r from-yellow-400 to-orange-500 bg-clip-text text-transparent"
              animate={{ scale: [1, 1.02, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              Pro
            </motion.span>
          </motion.h1>
        </motion.div>

        {/* Subtitle */}
        <motion.p
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="text-xl md:text-2xl text-gray-300 mb-12 max-w-3xl mx-auto leading-relaxed"
        >
          Revolutionary AI-powered cricket score prediction that outperforms traditional DLS methods.
          <motion.span 
            className="block mt-2 text-lg text-blue-300"
            animate={{ opacity: [0.7, 1, 0.7] }}
            transition={{ duration: 2, repeat: Infinity }}
          >
            7.86 RMSE • 96.45% Accuracy • Real-time Predictions
          </motion.span>
        </motion.p>

        {/* Cricket Ball Animation */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
          className="mb-16 h-16 relative"
        >
          <CricketBall />
        </motion.div>

        {/* Stats */}
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 1 }}
          className="grid grid-cols-3 gap-8 md:gap-16 max-w-2xl mx-auto"
        >
          <StatsCounter end={25000} label="Predictions Made" />
          <StatsCounter end={96} label="Accuracy %" />
          <StatsCounter end={15} label="Years of Data" />
        </motion.div>

        {/* Scroll Indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 2 }}
          className="absolute bottom-8 left-1/2 transform -translate-x-1/2"
        >
          <motion.div
            animate={{ y: [0, 10, 0] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="w-6 h-10 border-2 border-white/30 rounded-full flex justify-center"
          >
            <motion.div
              animate={{ y: [0, 12, 0] }}
              transition={{ duration: 2, repeat: Infinity }}
              className="w-1 h-3 bg-white/60 rounded-full mt-2"
            />
          </motion.div>
        </motion.div>
      </div>
    </section>
  )
}