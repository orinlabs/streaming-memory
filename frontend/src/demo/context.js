import { SYSTEM_PROMPT } from './config';

function makeSection(title, lines) {
  return ['[' + title + ']'].concat(lines).concat(['']);
}

function sanitizeBlock(text, fallback) {
  if (!text || !text.trim()) {
    return [fallback];
  }

  return text.replace(/\r/g, '').split('\n');
}

function buildConversationLines(history, userMessage) {
  const lines = [];

  history.slice(-6).forEach((message) => {
    const role = message.role === 'assistant' ? 'Assistant' : 'User';
    const contentLines = sanitizeBlock(message.content, '[empty message]');

    contentLines.forEach((line, index) => {
      if (index === 0) {
        lines.push(role + ': ' + line);
      } else {
        lines.push('  ' + line);
      }
    });
  });

  if (userMessage) {
    const contentLines = sanitizeBlock(userMessage, '[waiting for user prompt]');

    contentLines.forEach((line, index) => {
      if (index === 0) {
        lines.push('User: ' + line);
      } else {
        lines.push('  ' + line);
      }
    });
  } else {
    lines.push('[waiting for user prompt]');
  }

  return lines;
}

function buildWorkingMemoryLines(currentMemories) {
  if (!currentMemories.length) {
    return ['- [retrieval pending]'];
  }

  return currentMemories.map((memory) => '- ' + memory);
}

export function buildContextSnapshot({
  history,
  userMessage,
  currentMemories,
}) {
  const lines = []
    .concat(makeSection('SYSTEM', sanitizeBlock(SYSTEM_PROMPT, '[missing system prompt]')))
    .concat(makeSection('WORKING MEMORY', buildWorkingMemoryLines(currentMemories)))
    .concat(makeSection('CONVERSATION', buildConversationLines(history, userMessage)));

  return {
    lines,
    text: lines.join('\n'),
  };
}
