import { AnimatePresence, motion } from 'framer-motion';

function RangeControl({ label, value, min, max, step, suffix, disabled, onChange }) {
  return (
    <div className="flex items-center justify-between gap-4">
      <span className="text-xs text-[#666]">{label}</span>
      <div className="flex items-center gap-3">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          disabled={disabled}
          onChange={(event) => onChange(Number(event.target.value))}
          className="w-16 h-1 bg-[#e5e5e5] rounded-full appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:bg-[#666] [&::-webkit-slider-thumb]:rounded-full disabled:opacity-50"
        />
        <span className="w-20 text-right text-xs text-[#999] font-mono">
          {String(value) + suffix}
        </span>
      </div>
    </div>
  );
}

export default function SettingsPanel({
  open,
  isStreaming,
  updateFrequency,
  setUpdateFrequency,
  maxMemories,
  setMaxMemories,
  lookbackTokens,
  setLookbackTokens,
}) {
  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -10 }}
          className="absolute right-0 top-8 z-20 w-72 rounded-lg border border-[#eee] bg-white p-4 shadow-lg space-y-4"
        >
          <RangeControl
            label="Update frequency"
            value={updateFrequency}
            min="1"
            max="20"
            suffix={updateFrequency === 1 ? ' token' : ' tokens'}
            disabled={isStreaming}
            onChange={setUpdateFrequency}
          />
          <RangeControl
            label="Max memories"
            value={maxMemories}
            min="1"
            max="15"
            suffix=""
            disabled={isStreaming}
            onChange={setMaxMemories}
          />
          <RangeControl
            label="Lookback window"
            value={lookbackTokens}
            min="10"
            max="150"
            step="10"
            suffix=" tokens"
            disabled={isStreaming}
            onChange={setLookbackTokens}
          />
        </motion.div>
      )}
    </AnimatePresence>
  );
}
