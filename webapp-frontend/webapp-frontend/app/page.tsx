'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import HeroSection from '../components/HeroSection'
import PredictionForm from '../components/PredictionForm'
import ResultsDisplay from '../components/ResultsDisplay'
import { PredictionResult } from '../types/cricket'

export default function Home() {
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  return (
    <main className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Background Effects */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -inset-10 opacity-50">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-400 rounded-full mix-blend-multiply filter blur-xl animate-float"></div>
          <div className="absolute top-3/4 right-1/4 w-96 h-96 bg-purple-400 rounded-full mix-blend-multiply filter blur-xl animate-float" style={{ animationDelay: '2s' }}></div>
          <div className="absolute bottom-1/4 left-1/2 w-96 h-96 bg-pink-400 rounded-full mix-blend-multiply filter blur-xl animate-float" style={{ animationDelay: '4s' }}></div>
        </div>
      </div>

      {/* Content */}
      <div className="relative z-10">
        <HeroSection />
        
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="container mx-auto px-4 py-16"
        >
          <div className="grid lg:grid-cols-2 gap-12 items-start">
            {/* Prediction Form */}
            <motion.div
              initial={{ opacity: 0, x: -50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.6 }}
            >
              <PredictionForm 
                onPrediction={setPrediction}
                isLoading={isLoading}
                setIsLoading={setIsLoading}
              />
            </motion.div>

            {/* Results Display */}
            <motion.div
              initial={{ opacity: 0, x: 50 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.8 }}
            >
              <ResultsDisplay 
                prediction={prediction}
                isLoading={isLoading}
              />
            </motion.div>
          </div>
        </motion.div>

        {/* Features Section */}
        <motion.section
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="py-24 px-4"
        >
          <div className="container mx-auto">
            <motion.h2 
              className="text-4xl md:text-5xl font-bold text-center text-white mb-16"
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              viewport={{ once: true }}
            >
              Why CricPredict Pro?
            </motion.h2>
            
            <div className="grid md:grid-cols-3 gap-8">
              {[
                {
                  title: "AI-Powered Accuracy",
                  description: "7.86 RMSE - Better than traditional DLS methods",
                  icon: "🧠",
                  delay: 0
                },
                {
                  title: "Real-Time Predictions",
                  description: "Advanced RNN model with 19 enhanced features",
                  icon: "⚡",
                  delay: 0.2
                },
                {
                  title: "Professional Grade",
                  description: "Trained on 15+ years of IPL match data",
                  icon: "🏆",
                  delay: 0.4
                }
              ].map((feature, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 50 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.6, delay: feature.delay }}
                  viewport={{ once: true }}
                  whileHover={{ scale: 1.05, y: -10 }}
                  className="group cursor-pointer"
                >
                  <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 h-full border border-white/20 group-hover:bg-white/20 transition-all duration-300">
                    <div className="text-4xl mb-4 group-hover:scale-110 transition-transform duration-300">
                      {feature.icon}
                    </div>
                    <h3 className="text-xl font-semibold text-white mb-4">
                      {feature.title}
                    </h3>
                    <p className="text-gray-300 leading-relaxed">
                      {feature.description}
                    </p>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        </motion.section>
      </div>
    </main>
  )
}