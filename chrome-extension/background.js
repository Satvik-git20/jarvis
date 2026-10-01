// Clicking the toolbar icon opens the side panel instead of a popup.
chrome.sidePanel
  .setPanelBehavior({ openPanelOnActionClick: true })
  .catch(() => {});
