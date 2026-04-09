import {
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';

const RANDOM_CHARSET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-:/.() ';

function getRandomChar(targetChar) {
  if (targetChar === ' ') {
    return ' ';
  }

  const index = Math.floor(Math.random() * RANDOM_CHARSET.length);
  return RANDOM_CHARSET[index];
}

function getDiffSlices(fromText, toText) {
  let prefixLength = 0;
  let suffixLength = 0;

  while (
    prefixLength < fromText.length &&
    prefixLength < toText.length &&
    fromText[prefixLength] === toText[prefixLength]
  ) {
    prefixLength += 1;
  }

  while (
    suffixLength < fromText.length - prefixLength &&
    suffixLength < toText.length - prefixLength &&
    fromText[fromText.length - 1 - suffixLength] === toText[toText.length - 1 - suffixLength]
  ) {
    suffixLength += 1;
  }

  return {
    prefix: toText.slice(0, prefixLength),
    suffix: suffixLength > 0 ? toText.slice(toText.length - suffixLength) : '',
    toMiddle: toText.slice(prefixLength, toText.length - suffixLength),
  };
}

function ShuffleLine({ fromText, toText, animate }) {
  const slices = useMemo(() => getDiffSlices(fromText, toText), [fromText, toText]);
  const [displayMiddle, setDisplayMiddle] = useState(slices.toMiddle);

  useEffect(() => {
    if (!animate || fromText === toText) {
      setDisplayMiddle(slices.toMiddle);
      return undefined;
    }

    let frame = 0;
    const totalFrames = 12;

    const intervalId = window.setInterval(() => {
      frame += 1;
      const progress = Math.min(frame / totalFrames, 1);

      setDisplayMiddle(
        slices.toMiddle
          .split('')
          .map((char) =>
            char === ' ' ? ' ' : Math.random() < progress ? char : getRandomChar(char)
          )
          .join('')
      );

      if (progress >= 1) {
        window.clearInterval(intervalId);
      }
    }, 28);

    return () => window.clearInterval(intervalId);
  }, [animate, fromText, slices, toText]);

  return (
    <span>
      <span>{slices.prefix}</span>
      <span className={animate ? 'text-[#1a1a1a] font-medium' : ''}>{displayMiddle}</span>
      <span>{slices.suffix}</span>
    </span>
  );
}

export default function TerminalContextPanel({
  lines,
  activeRevisionId,
  revisionCount,
  generatedTokens,
  contextTokenCount,
  currentMemories,
  uniqueMemoriesCount,
  memoryUpdates,
  allMemoriesTokens,
}) {
  const [animatingRevisionId, setAnimatingRevisionId] = useState(0);
  const settledLinesRef = useRef(lines);
  const previousLinesRef = useRef(lines);
  const latestRevisionRef = useRef(activeRevisionId);

  useEffect(() => {
    if (
      animatingRevisionId === 0 &&
      latestRevisionRef.current === activeRevisionId
    ) {
      settledLinesRef.current = lines;
    }
  }, [activeRevisionId, animatingRevisionId, lines]);

  useEffect(() => {
    if (!activeRevisionId || latestRevisionRef.current === activeRevisionId) {
      latestRevisionRef.current = activeRevisionId;
      return undefined;
    }

    previousLinesRef.current = settledLinesRef.current;
    setAnimatingRevisionId(activeRevisionId);
    latestRevisionRef.current = activeRevisionId;

    const timeoutId = window.setTimeout(() => {
      settledLinesRef.current = lines;
      setAnimatingRevisionId(0);
    }, 420);

    return () => window.clearTimeout(timeoutId);
  }, [activeRevisionId, lines]);

  return (
    <div className="flex h-full flex-col">
      <div className="mb-4 flex items-baseline justify-between gap-4">
        <div>
          <h2 className="text-sm font-semibold text-[#1a1a1a]">Current model input</h2>
          <p className="mt-1 text-xs text-[#999]">
            The engine rewrites this prompt while the model generates.
          </p>
        </div>
        <div className="flex items-center gap-4 text-xs text-[#999]">
          <span className="font-mono">
            rev {String(revisionCount || 0).padStart(2, '0')}
          </span>
          <span className="font-mono">
            {contextTokenCount ? contextTokenCount.toLocaleString() + ' tok' : 'warming up'}
          </span>
        </div>
      </div>

      <div className="relative min-h-0 flex-1 overflow-y-auto rounded-lg border border-[#eee] bg-[#fafafa]">
        <div className="grid grid-cols-[48px_minmax(0,1fr)] gap-0 px-4 py-4 font-mono text-[13px] leading-7 text-[#444]">
          <div className="select-none border-r border-[#eee] pr-3 text-right text-[#ccc]">
            {lines.map((_, index) => (
              <div key={index}>{index + 1}</div>
            ))}
          </div>
          <div className="pl-4">
            {lines.map((line, index) => {
              const previousLine = previousLinesRef.current[index] || '';
              const isChanged =
                animatingRevisionId === activeRevisionId && previousLine !== line;

              return (
                <div
                  key={index + '-' + line}
                  className={[
                    'whitespace-pre-wrap break-words rounded px-1',
                    line.startsWith('[')
                      ? 'text-[11px] font-medium uppercase tracking-[0.2em] text-[#999]'
                      : '',
                    isChanged ? 'bg-[#fff8e0]' : '',
                  ].join(' ')}
                >
                  <ShuffleLine
                    fromText={previousLine}
                    toText={line}
                    animate={isChanged}
                  />
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <div className="mt-4 flex items-center gap-6 text-xs text-[#999]">
        <span>{memoryUpdates} rewrites</span>
        <span>{currentMemories.length} active</span>
        <span>{uniqueMemoriesCount} touched</span>
        <span>{generatedTokens} generated</span>
        <span className="ml-auto font-mono text-[#666]">
          {contextTokenCount.toLocaleString()} / {allMemoriesTokens.toLocaleString()} tokens
        </span>
      </div>
    </div>
  );
}
