import { useEffect, useRef, useState, type KeyboardEvent, type ReactNode, type Ref } from 'react';
import { copyText } from '../store';

const PATHS = {
  folder: 'M3 7.5A2.5 2.5 0 0 1 5.5 5H9l2 2h7.5A2.5 2.5 0 0 1 21 9.5v7a2.5 2.5 0 0 1-2.5 2.5h-13A2.5 2.5 0 0 1 3 16.5z',
  clock: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM12 7.5V12l3 2',
  download: 'M12 4v11M7 10l5 5 5-5M5 20h14',
  x: 'M6 6l12 12M18 6 6 18',
  panelLeft: 'M5 4h14a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2zM9.5 4v16',
  panelRight: 'M5 4h14a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2zM14.5 4v16',
  sun: 'M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8zM12 2v2.5M12 19.5V22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M2 12h2.5M19.5 12H22M4.9 19.1l1.8-1.8M17.3 6.7l1.8-1.8',
  moon: 'M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z',
  monitor: 'M5 4h14a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2zM8.5 20h7M12 16v4',
  globe: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM3 12h18M12 3c2.5 2.6 3.8 5.6 3.8 9s-1.3 6.4-3.8 9M12 3c-2.5 2.6-3.8 5.6-3.8 9s1.3 6.4 3.8 9',
  help: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM9.6 9.3a2.5 2.5 0 1 1 3.4 2.3c-.6.3-1 .9-1 1.6v.3M12 17h.01',
  search: 'M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14zM20 20l-4-4',
  fit: 'M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5',
  plus: 'M12 5v14M5 12h14',
  minus: 'M5 12h14',
  target: 'M12 4a8 8 0 1 0 0 16 8 8 0 0 0 0-16zM12 10a2 2 0 1 0 0 4 2 2 0 0 0 0-4zM12 1.5V4M12 20v2.5M1.5 12H4M20 12h2.5',
  chevronRight: 'M9 6l6 6-6 6',
  chevronDown: 'M6 9l6 6 6-6',
  external: 'M14 4h6v6M20 4l-9 9M18 14v5a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h5',
  copy: 'M10 8h9a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2h-9a2 2 0 0 1-2-2v-9a2 2 0 0 1 2-2zM16 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h3',
  eye: 'M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12zM12 9a3 3 0 1 0 0 6 3 3 0 0 0 0-6z',
  layers: 'M12 3l9 5-9 5-9-5zM3 13l9 5 9-5',
  alert: 'M12 3.5 21.5 20h-19zM12 10v4.5M12 17.5h.01',
  info: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM12 11v6M12 7.5h.01',
  error: 'M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zM9 9l6 6M15 9l-6 6',
  check: 'M5 12.5l4.5 4.5L19 7',
  arrowLeft: 'M19 12H5M11 6l-6 6 6 6',
  arrowUp: 'M12 19V5M6 11l6-6 6 6',
  arrowDown: 'M12 5v14M6 13l6 6 6-6',
  play: 'M7 5l12 7-12 7z',
  graph: 'M6 3.5a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5zM18 3.5a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5zM12 15.5a2.5 2.5 0 1 0 0 5 2.5 2.5 0 0 0 0-5zM7.3 8.2l3.5 7.5M16.7 8.2l-3.5 7.5M8.5 6h7',
  trace: 'M3 12h4l3-7 4 14 3-7h4',
  filter: 'M4 5h16l-6.2 7.5V18l-3.6 2v-7.5z',
  table: 'M5 4h14a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2zM3 10h18M9 10v10',
  cube: 'M12 3l8 4.5v9L12 21l-8-4.5v-9zM12 12l8-4.5M12 12v9M12 12 4 7.5',
  refresh: 'M20 11a8 8 0 1 0-2.4 5.7M20 4.5V11h-6.5',
} as const;

export type IconName = keyof typeof PATHS;

export function Icon({ name, size = 16, className }: { name: IconName; size?: number; className?: string }) {
  return (
    <svg
      className={'icon' + (className ? ` ${className}` : '')}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.9}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d={PATHS[name]} />
    </svg>
  );
}

export function IconButton({
  icon,
  title,
  onClick,
  active,
  disabled,
  size = 16,
  className,
}: {
  icon: IconName;
  title: string;
  onClick?: () => void;
  active?: boolean;
  disabled?: boolean;
  size?: number;
  className?: string;
}) {
  return (
    <button
      type="button"
      className={'icon-btn' + (active ? ' active' : '') + (className ? ` ${className}` : '')}
      title={title}
      aria-label={title}
      aria-pressed={active}
      onClick={onClick}
      disabled={disabled}
    >
      <Icon name={icon} size={size} />
    </button>
  );
}

export function Button({
  children,
  icon,
  onClick,
  kind = 'default',
  disabled,
  title,
  small,
}: {
  children?: ReactNode;
  icon?: IconName;
  onClick?: () => void;
  kind?: 'primary' | 'default' | 'ghost';
  disabled?: boolean;
  title?: string;
  small?: boolean;
}) {
  return (
    <button
      type="button"
      className={`btn btn-${kind}${small ? ' btn-sm' : ''}`}
      onClick={onClick}
      disabled={disabled}
      title={title}
    >
      {icon && <Icon name={icon} size={small ? 14 : 15} />}
      {children && <span>{children}</span>}
    </button>
  );
}

export function Section({
  title,
  count,
  children,
  defaultOpen = true,
  actions,
}: {
  title: ReactNode;
  count?: number | string;
  children: ReactNode;
  defaultOpen?: boolean;
  actions?: ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className={'section' + (open ? ' open' : '')}>
      <div className="section-head" onClick={() => setOpen(!open)}>
        <Icon name={open ? 'chevronDown' : 'chevronRight'} size={14} />
        <span className="section-title">{title}</span>
        {count !== undefined && <span className="section-count">{count}</span>}
        {actions && (
          <span className="section-actions" onClick={(e) => e.stopPropagation()}>
            {actions}
          </span>
        )}
      </div>
      {open && <div className="section-body">{children}</div>}
    </div>
  );
}

export function Badge({ children, tone = 'neutral' }: { children: ReactNode; tone?: 'neutral' | 'accent' | 'error' | 'warning' | 'info' | 'good' }) {
  return <span className={`badge badge-${tone}`}>{children}</span>;
}

export function Dot({ color }: { color: string }) {
  return <span className="dot" style={{ background: color }} />;
}

export function CopyButton({ text, title }: { text: string; title?: string }) {
  return <IconButton icon="copy" size={13} title={title ?? 'Copy'} className="copy-btn" onClick={() => void copyText(text)} />;
}

export function Empty({ icon = 'info', title, children }: { icon?: IconName; title: ReactNode; children?: ReactNode }) {
  return (
    <div className="empty">
      <Icon name={icon} size={28} />
      <div className="empty-title">{title}</div>
      {children && <div className="empty-body">{children}</div>}
    </div>
  );
}

export function Spinner({ size = 16 }: { size?: number }) {
  return <span className="spinner" style={{ width: size, height: size }} />;
}

export function KV({ k, children, mono }: { k: ReactNode; children: ReactNode; mono?: boolean }) {
  return (
    <div className="kv">
      <div className="kv-k">{k}</div>
      <div className={'kv-v' + (mono ? ' mono' : '')}>{children}</div>
    </div>
  );
}

export interface MenuItem {
  label?: string;
  icon?: IconName;
  onClick?: () => void;
  checked?: boolean;
  disabled?: boolean;
  separator?: boolean;
  hint?: string;
}

export function MenuList({ items, onClose }: { items: MenuItem[]; onClose: () => void }) {
  return (
    <div className="menu" role="menu">
      {items.map((it, i) =>
        it.separator ? (
          <div key={i} className="menu-sep" />
        ) : (
          <button
            key={i}
            type="button"
            role="menuitem"
            className="menu-item"
            disabled={it.disabled}
            onClick={() => {
              onClose();
              it.onClick?.();
            }}
          >
            <span className="menu-check">{it.checked ? <Icon name="check" size={14} /> : it.icon ? <Icon name={it.icon} size={14} /> : null}</span>
            <span className="menu-label">{it.label}</span>
            {it.hint && <span className="menu-hint">{it.hint}</span>}
          </button>
        ),
      )}
    </div>
  );
}

/** Button that opens a menu below it; closes on outside click or Escape. */
export function Dropdown({
  icon,
  label,
  title,
  items,
  align = 'left',
  disabled,
}: {
  icon?: IconName;
  label?: string;
  title: string;
  items: MenuItem[] | (() => MenuItem[]);
  align?: 'left' | 'right';
  disabled?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    window.addEventListener('mousedown', onDown);
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('mousedown', onDown);
      window.removeEventListener('keydown', onKey);
    };
  }, [open]);
  const list = typeof items === 'function' ? (open ? items() : []) : items;
  return (
    <div className="dropdown" ref={ref}>
      <button
        type="button"
        className={'btn btn-ghost' + (open ? ' active' : '')}
        title={title}
        disabled={disabled}
        onClick={() => setOpen(!open)}
      >
        {icon && <Icon name={icon} size={15} />}
        {label && <span>{label}</span>}
        <Icon name="chevronDown" size={12} className="caret" />
      </button>
      {open && (
        <div className={'dropdown-pop ' + align}>
          <MenuList items={list} onClose={() => setOpen(false)} />
        </div>
      )}
    </div>
  );
}

export function Segmented<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div className="segmented" role="tablist">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="tab"
          aria-selected={o.value === value}
          className={o.value === value ? 'on' : ''}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function SearchInput({
  value,
  onChange,
  placeholder,
  inputRef,
  onKeyDown,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder: string;
  inputRef?: Ref<HTMLInputElement>;
  onKeyDown?: (e: KeyboardEvent<HTMLInputElement>) => void;
}) {
  return (
    <div className="search-input">
      <Icon name="search" size={14} />
      <input
        ref={inputRef}
        value={value}
        placeholder={placeholder}
        spellCheck={false}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={onKeyDown}
      />
      {value && <IconButton icon="x" size={12} title="Clear" onClick={() => onChange('')} />}
    </div>
  );
}
