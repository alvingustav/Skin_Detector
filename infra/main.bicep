@description('The location used for all deployed resources')
param location string = resourceGroup().location

@description('The name of the App Service app.')
param webServiceName string = ''

// Load abbreviations and generate unique resource token
var abbrs = loadJsonContent('abbreviations.json')
var resourceToken = toLower(uniqueString(subscription().id, resourceGroup().id, location))
var tags = { 'azd-env-name': resourceToken }

// App Service Plan - Upgrade ke B1 untuk performa lebih baik
resource appServicePlan 'Microsoft.Web/serverfarms@2022-03-01' = {
  name: '${abbrs.webServerFarms}${resourceToken}'
  location: location
  tags: tags
  sku: {
    name: 'B1'  // Upgrade dari F1 ke B1 untuk memory dan CPU lebih besar
    tier: 'Basic'
  }
  kind: 'linux'
  properties: {
    reserved: true
  }
}

// App Service dengan konfigurasi yang dioptimalkan
resource appService 'Microsoft.Web/sites@2022-03-01' = {
  name: !empty(webServiceName) ? webServiceName : '${abbrs.webSitesAppService}web-${resourceToken}'
  location: location
  tags: union(tags, { 'azd-service-name': 'web' })
  properties: {
    serverFarmId: appServicePlan.id
    siteConfig: {
      linuxFxVersion: 'PYTHON|3.11'
      alwaysOn: true  // Enable untuk B1 tier
      // ✅ PERBAIKAN: Startup command yang benar untuk Flask-SocketIO
      appCommandLine: 'python -m gunicorn --bind=0.0.0.0:8000 --timeout 600 --worker-class gevent --workers 1 app:app'
      appSettings: [
        {
          name: 'SCM_DO_BUILD_DURING_DEPLOYMENT'
          value: 'true'
        }
        {
          name: 'WEBSITES_ENABLE_APP_SERVICE_STORAGE'
          value: 'false'
        }
        {
          name: 'MODEL_PATH'
          value: 'my_model1.pt'  // Model custom Anda
        }
        {
          name: 'WEBSITES_PORT'
          value: '8000'
        }
        {
          name: 'SCM_COMMAND_IDLE_TIMEOUT'
          value: '1800'  // 30 menit timeout untuk build
        }
        {
          name: 'WEBSITE_TIME_ZONE'
          value: 'Asia/Jakarta'
        }
        {
          name: 'PYTHONPATH'
          value: '/home/site/wwwroot'  // Pastikan Python dapat menemukan module app
        }
        {
          name: 'FLASK_APP'
          value: 'app.py'
        }
        {
          name: 'FLASK_ENV'
          value: 'production'
        }
      ]
      // ✅ TAMBAHAN: Konfigurasi untuk debugging
      detailedErrorLoggingEnabled: true
      httpLoggingEnabled: true
      logsDirectorySizeLimit: 35
    }
  }
}

// Output values
output AZURE_LOCATION string = location
output AZURE_TENANT_ID string = tenant().tenantId
output WEBAPP_NAME string = appService.name
output WEBAPP_URI string = 'https://${appService.properties.defaultHostName}'