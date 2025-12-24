/**
 * Hotel CCTV Monitoring System - Main JavaScript
 * Common functions and utilities
 */

// CSRF Token helper for AJAX requests
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

const csrfToken = getCookie('csrftoken');

// Default fetch options with CSRF
function fetchWithCSRF(url, options = {}) {
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': csrfToken,
        },
    };
    
    return fetch(url, { ...defaultOptions, ...options });
}

// Sidebar toggle functionality
document.addEventListener('DOMContentLoaded', function() {
    const sidebar = document.querySelector('.sidebar');
    const layout = document.querySelector('.layout');
    const collapseBtn = document.querySelector('.collapse-btn');
    const mobileMenuBtn = document.querySelector('#mobileMenuBtn');
    const mobileBackdrop = document.querySelector('#mobileBackdrop');
    
    // Mobile menu toggle
    if (mobileMenuBtn && sidebar) {
        const toggleMobileMenu = function(show) {
            if (show) {
                sidebar.classList.add('mobile-open');
                mobileMenuBtn.classList.add('active');
                if (mobileBackdrop) mobileBackdrop.classList.add('active');
                document.body.style.overflow = 'hidden'; // Prevent scroll
            } else {
                sidebar.classList.remove('mobile-open');
                mobileMenuBtn.classList.remove('active');
                if (mobileBackdrop) mobileBackdrop.classList.remove('active');
                document.body.style.overflow = ''; // Restore scroll
            }
        };
        
        mobileMenuBtn.addEventListener('click', function(e) {
            e.stopPropagation();
            const isOpen = sidebar.classList.contains('mobile-open');
            toggleMobileMenu(!isOpen);
        });
        
        // Close when clicking backdrop
        if (mobileBackdrop) {
            mobileBackdrop.addEventListener('click', function() {
                toggleMobileMenu(false);
            });
        }
        
        // Close when clicking nav links on mobile
        sidebar.querySelectorAll('.nav-item:not(.nav-parent)').forEach(link => {
            link.addEventListener('click', function() {
                if (window.innerWidth <= 768) {
                    toggleMobileMenu(false);
                }
            });
        });
    }
    
    // Desktop sidebar collapse - only on desktop
    const isMobile = window.innerWidth <= 768;
    if (!isMobile && collapseBtn && sidebar && layout) {
        // Load saved state from localStorage - default to COLLAPSED for compact view
        const savedState = localStorage.getItem('sidebarCollapsed');
        const isCollapsed = savedState === null ? true : savedState === 'true';
        if (isCollapsed) {
            sidebar.classList.add('collapsed');
            layout.classList.add('layout-collapsed');
            collapseBtn.textContent = '→';
        }
    
    if (collapseBtn) {
        collapseBtn.addEventListener('click', function() {
            sidebar.classList.toggle('collapsed');
            layout.classList.toggle('layout-collapsed');
            
            // Save state to localStorage
            localStorage.setItem('sidebarCollapsed', sidebar.classList.contains('collapsed'));
            
            // Update button text
            this.textContent = sidebar.classList.contains('collapsed') ? '→' : '←';
        });
    }
    
    // Sub-navigation toggle
    const navParents = document.querySelectorAll('.nav-parent');
    navParents.forEach(parent => {
        parent.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            
            const navGroup = this.closest('.nav-group');
            const sublist = navGroup?.querySelector('.nav-sublist');
            const caret = this.querySelector('.caret');
            
            if (sublist) {
                // Check current display state - handle both '' and 'none' as hidden
                const isHidden = !sublist.style.display || sublist.style.display === 'none';
                sublist.style.display = isHidden ? 'flex' : 'none';
                if (caret) {
                    caret.classList.toggle('open', isHidden);
                }
            }
        });
    });
    
    // Initialize sublists that should be visible
    document.querySelectorAll('.nav-sublist').forEach(sublist => {
        // If it doesn't have explicit display style set, hide it by default
        // unless it has items that are active
        const hasActiveItem = sublist.querySelector('.active');
        if (!sublist.style.display && !hasActiveItem) {
            sublist.style.display = 'none';
        }
    });
});

// Format date for display
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('ko-KR', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
    });
}

// Format time only
function formatTime(dateString) {
    const date = new Date(dateString);
    return date.toLocaleTimeString('ko-KR', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
}

// Modal functions
function openModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.style.display = 'flex';
    }
}

function closeModal(modalId) {
    const modal = document.getElementById(modalId);
    if (modal) {
        modal.style.display = 'none';
    }
}

// Close modal when clicking backdrop
document.addEventListener('click', function(e) {
    if (e.target.classList.contains('modal-backdrop')) {
        e.target.style.display = 'none';
    }
});

// Form validation helpers
function validateRequired(formData, fields) {
    const errors = [];
    fields.forEach(field => {
        if (!formData.get(field) || formData.get(field).trim() === '') {
            errors.push(`${field} is required`);
        }
    });
    return errors;
}

// Toast notification (simple implementation)
function showToast(message, type = 'info') {
    // Remove existing toasts
    const existingToasts = document.querySelectorAll('.toast');
    existingToasts.forEach(toast => toast.remove());
    
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    toast.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        padding: 12px 20px;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        z-index: 9999;
        animation: slideIn 0.3s ease;
        background: ${type === 'success' ? '#2f9f57' : type === 'error' ? '#e15b5b' : '#1c1373'};
    `;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Add CSS for toast animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Confirmation dialog
function confirmAction(message) {
    return new Promise((resolve) => {
        const result = window.confirm(message);
        resolve(result);
    });
}

// Debounce function for search inputs
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Simple pie chart drawing function (Canvas-based)
function drawPieChart(canvasId, data, colors) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = Math.min(centerX, centerY) - 10;
    
    const total = data.reduce((sum, item) => sum + item.value, 0);
    let startAngle = -Math.PI / 2;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw slices
    data.forEach((item, index) => {
        const sliceAngle = (item.value / total) * 2 * Math.PI;
        
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.arc(centerX, centerY, radius, startAngle, startAngle + sliceAngle);
        ctx.closePath();
        
        ctx.fillStyle = colors[index % colors.length];
        ctx.fill();
        
        startAngle += sliceAngle;
    });
    
    // Draw center hole (donut chart)
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius * 0.5, 0, 2 * Math.PI);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
}

// Simple line chart drawing function (SVG-based)
function drawLineChart(containerId, data, color = '#1c1373') {
    const container = document.getElementById(containerId);
    if (!container || !data.length) return;
    
    const width = container.clientWidth;
    const height = container.clientHeight || 180;
    const padding = { top: 20, right: 20, bottom: 30, left: 40 };
    
    const maxValue = Math.max(...data.map(d => d.value)) || 1;
    const chartWidth = width - padding.left - padding.right;
    const chartHeight = height - padding.top - padding.bottom;
    
    // Generate path
    const points = data.map((d, i) => {
        const x = padding.left + (i / (data.length - 1)) * chartWidth;
        const y = padding.top + chartHeight - (d.value / maxValue) * chartHeight;
        return `${x},${y}`;
    });
    
    const pathD = `M ${points.join(' L ')}`;
    
    // Create SVG
    const svg = `
        <svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
            <path d="${pathD}" fill="none" stroke="${color}" stroke-width="2"/>
            ${data.map((d, i) => {
                const x = padding.left + (i / (data.length - 1)) * chartWidth;
                const y = padding.top + chartHeight - (d.value / maxValue) * chartHeight;
                return `<circle cx="${x}" cy="${y}" r="4" fill="${color}"/>`;
            }).join('')}
        </svg>
    `;
    
    container.innerHTML = svg;
}

// Export functions for global use
window.getCookie = getCookie;
window.fetchWithCSRF = fetchWithCSRF;
window.formatDate = formatDate;
window.formatTime = formatTime;
window.openModal = openModal;
window.closeModal = closeModal;
window.showToast = showToast;
window.confirmAction = confirmAction;
window.debounce = debounce;
window.drawPieChart = drawPieChart;
window.drawLineChart = drawLineChart;
