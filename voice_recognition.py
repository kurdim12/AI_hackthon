import speech_recognition as sr
import pyttsx3
from googletrans import Translator
import streamlit as st
import numpy as np
import io
import wave
import pyaudio
from typing import Dict, Any, Optional
import openai
import os
from ai_agent import FinancialAnalysisAgent

class VoiceAssistant:
    """Voice-enabled assistant with Arabic support"""
    
    def __init__(self, df, language='ar'):
        self.df = df
        self.language = language
        self.recognizer = sr.Recognizer()
        self.translator = Translator()
        self.agent = FinancialAnalysisAgent(df)
        self.tts_engine = pyttsx3.init()
        
        # Configure TTS for Arabic
        voices = self.tts_engine.getProperty('voices')
        for voice in voices:
            if 'arabic' in voice.name.lower() or 'ar' in voice.id.lower():
                self.tts_engine.setProperty('voice', voice.id)
                break
        
        # Adjust speed for Arabic
        self.tts_engine.setProperty('rate', 150)
        
        # Language settings
        self.languages = {
            'ar': {'name': 'العربية', 'code': 'ar-SA'},
            'en': {'name': 'English', 'code': 'en-US'}
        }
    
    def record_audio(self, duration=5) -> Optional[sr.AudioData]:
        """Record audio from microphone"""
        try:
            with sr.Microphone() as source:
                st.write("🎤 يرجى التحدث...")  # Please speak...
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = self.recognizer.listen(source, timeout=duration)
                return audio
        except Exception as e:
            st.error(f"خطأ في التسجيل: {str(e)}")  # Recording error
            return None
    
    def recognize_speech(self, audio: sr.AudioData, language='ar') -> Optional[str]:
        """Convert speech to text"""
        try:
            # Use Google Speech Recognition
            lang_code = self.languages[language]['code']
            text = self.recognizer.recognize_google(audio, language=lang_code)
            return text
        except sr.UnknownValueError:
            st.warning("لم أتمكن من فهم الكلام")  # Could not understand speech
            return None
        except sr.RequestError as e:
            st.error(f"خطأ في خدمة التعرف على الكلام: {str(e)}")
            return None
    
    def translate_text(self, text: str, target_language: str = 'en') -> str:
        """Translate text between Arabic and English"""
        try:
            if target_language == 'ar':
                result = self.translator.translate(text, src='en', dest='ar')
            else:
                result = self.translator.translate(text, src='ar', dest='en')
            return result.text
        except Exception as e:
            st.error(f"خطأ في الترجمة: {str(e)}")  # Translation error
            return text
    
    def speak_text(self, text: str, language: str = 'ar'):
        """Convert text to speech"""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            st.error(f"خطأ في النطق: {str(e)}")  # Speech error
    
    def process_voice_query(self, audio: sr.AudioData) -> Dict[str, Any]:
        """Process voice query through AI agent"""
        # Recognize speech in Arabic
        arabic_text = self.recognize_speech(audio, language='ar')
        if not arabic_text:
            return {'success': False, 'error': 'لم يتم التعرف على الكلام'}
        
        # Translate to English for AI processing
        english_text = self.translate_text(arabic_text, target_language='en')
        
        # Process with AI agent
        result = self.agent.query(english_text)
        
        if result['success']:
            # Translate response back to Arabic
            arabic_response = self.translate_text(result['response'], target_language='ar')
            
            return {
                'success': True,
                'original_query': arabic_text,
                'translated_query': english_text,
                'english_response': result['response'],
                'arabic_response': arabic_response
            }
        else:
            return {
                'success': False,
                'error': result.get('error', 'خطأ في معالجة الاستعلام')
            }
    
    def create_voice_interface(self):
        """Create Streamlit voice interface"""
        st.header("🎤 المساعد الصوتي - Voice Assistant")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("التحدث بالعربية - Speak in Arabic")
            
            # Language selection
            language = st.selectbox(
                "اختر اللغة - Select Language",
                options=['ar', 'en'],
                format_func=lambda x: self.languages[x]['name']
            )
            
            # Voice recording button
            if st.button("🎤 ابدأ التسجيل - Start Recording", key="voice_record"):
                with st.spinner("جاري التسجيل... Recording..."):
                    audio = self.record_audio()
                    
                    if audio:
                        st.success("تم التسجيل بنجاح! Recording complete!")
                        
                        # Process the audio
                        result = self.process_voice_query(audio)
                        
                        if result['success']:
                            st.write("**ما قلته (What you said):**")
                            st.info(result['original_query'])
                            
                            st.write("**الترجمة (Translation):**")
                            st.info(result['translated_query'])
                            
                            st.write("**الإجابة (Response):**")
                            st.success(result['arabic_response'])
                            
                            # Speak the response
                            if st.button("🔊 اقرأ الإجابة - Read Response"):
                                self.speak_text(result['arabic_response'], language='ar')
                        else:
                            st.error(result.get('error', 'حدث خطأ'))
        
        with col2:
            st.subheader("أمثلة للأسئلة - Example Questions")
            
            examples = {
                "ما هو معدل الفشل؟": "What is the failure rate?",
                "أي فرع لديه أكثر الإخفاقات؟": "Which branch has the most failures?",
                "أظهر لي أداء اليوم": "Show me today's performance",
                "ما هو تأثير الإيرادات؟": "What is the revenue impact?",
                "قارن بين جميع الفروع": "Compare all branches"
            }
            
            for arabic, english in examples.items():
                with st.expander(arabic):
                    st.write(f"**العربية:** {arabic}")
                    st.write(f"**English:** {english}")
                    
                    if st.button(f"🎤 جرب هذا السؤال", key=f"try_{arabic[:10]}"):
                        # Process this question
                        result = self.agent.query(english)
                        if result['success']:
                            arabic_response = self.translate_text(result['response'], target_language='ar')
                            st.success(arabic_response)
        
        # Voice commands help
        with st.expander("📖 تعليمات الاستخدام - Usage Instructions"):
            st.markdown("""
            ### كيفية استخدام المساعد الصوتي:
            
            1. **اضغط على زر التسجيل** - سيبدأ التسجيل لمدة 5 ثوان
            2. **تحدث بوضوح** - اطرح سؤالك باللغة العربية
            3. **انتظر المعالجة** - سيتم ترجمة سؤالك ومعالجته
            4. **اقرأ الإجابة** - ستظهر الإجابة بالعربية
            5. **استمع للإجابة** - يمكنك الضغط على زر القراءة
            
            ### How to use voice assistant:
            
            1. **Click record button** - Recording starts for 5 seconds
            2. **Speak clearly** - Ask your question in Arabic
            3. **Wait for processing** - Your question will be translated
            4. **Read the response** - Answer appears in Arabic
            5. **Listen to response** - Click read button for audio
            """)

# Arabic number formatting utility
def format_number_arabic(number: float) -> str:
    """Format numbers with Arabic numerals"""
    arabic_numerals = str.maketrans('0123456789', '٠١٢٣٤٥٦٧٨٩')
    return f"{number:,.2f}".translate(arabic_numerals)

# Arabic text utilities
def get_arabic_metrics(df) -> Dict[str, str]:
    """Get metrics with Arabic labels"""
    total_transactions = len(df)
    failed_transactions = df['is_failed'].sum()
    failure_rate = (failed_transactions / total_transactions) * 100
    
    return {
        'إجمالي المعاملات': format_number_arabic(total_transactions),
        'المعاملات الفاشلة': format_number_arabic(failed_transactions),
        'معدل الفشل': f"{format_number_arabic(failure_rate)}٪",
        'المبلغ المفقود': f"{format_number_arabic(df[df['is_failed']]['transaction_amount'].sum())} ريال"
    }

# Enhanced Streamlit interface with voice support
def create_voice_enabled_interface(df):
    """Create voice-enabled interface in Streamlit"""
    assistant = VoiceAssistant(df)
    assistant.create_voice_interface()