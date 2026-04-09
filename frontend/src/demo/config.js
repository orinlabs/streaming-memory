export const API_URL =
  import.meta.env.VITE_API_URL ||
  'https://orin-labs--streaming-memory-familyassistant-serve.modal.run';

export const SYSTEM_PROMPT = [
  'You are a helpful personal assistant who has access to the user\'s memories and notes.',
  '',
  'You help them think through decisions by drawing on what you know about their life, relationships, and past experiences.',
  '',
  'When memories are provided, use them naturally to inform your responses. Make connections between different memories when relevant.',
  '',
  'Think step by step in <think>...</think> tags before responding.',
  '',
  'Be warm and helpful, like a thoughtful friend who knows them well.',
  '',
  'Important: Do not use emojis in your responses.',
].join('\n');

export const DEMO_CONFIG = {
  name: 'Family Assistant',
  eyebrow: 'Streaming memory demo',
  headline: 'The engine keeps rewriting the model input during generation.',
  description:
    'Watch the context window change in-place while the answer keeps streaming. Each revision updates the next prompt the model sees.',
  memoryDescription:
    'Your assistant has access to ~280 memories spanning family, work, hobbies, and daily life.',
  placeholder: 'Ask for advice or help planning...',
  suggestedQuestions: [
    'What should I get my dad for his birthday?',
    'I\'m feeling anxious about work lately, any advice?',
    'Help me figure out what to do about my apartment lease',
    'What could I get mom for her birthday?',
  ],
};
