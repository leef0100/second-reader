// OHIF configuration: single data source = Orthanc B via the gateway (same origin).
window.config = {
  routerBasename: '/',
  showStudyList: true,
  investigationalUseDialog: { option: 'always' }, // OHIF's own "investigational use" banner
  whiteLabeling: {
    createLogoComponentFn: function (React) {
      return React.createElement('div', { style: { color: '#fbbf24', fontWeight: 600, fontSize: 12 } },
        'Second Reader — For research use only. Not for diagnostic use.');
    },
  },
  defaultDataSourceName: 'orthancB',
  dataSources: [
    {
      namespace: '@ohif/extension-default.dataSourcesModule.dicomweb',
      sourceName: 'orthancB',
      configuration: {
        friendlyName: 'Second Reader archive (de-identified)',
        name: 'orthancB',
        wadoUriRoot: '/pacs-b/wado',
        qidoRoot: '/pacs-b/dicom-web',
        wadoRoot: '/pacs-b/dicom-web',
        qidoSupportsIncludeField: false,
        supportsReject: false,
        imageRendering: 'wadors',
        thumbnailRendering: 'wadors',
        enableStudyLazyLoad: true,
        supportsFuzzyMatching: false,
        supportsWildcard: true,
        dicomUploadEnabled: false,
        bulkDataURI: { enabled: true },
      },
    },
  ],
};
