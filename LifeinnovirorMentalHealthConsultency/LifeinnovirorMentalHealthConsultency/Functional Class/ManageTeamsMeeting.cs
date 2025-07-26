using Microsoft.Graph;
using Microsoft.Graph.Models;
using Microsoft.Identity.Client;
using Microsoft.Kiota.Abstractions;
using Microsoft.Kiota.Abstractions.Authentication;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

public class TeamsMeetingHelper : IAuthenticationProvider
{
    private readonly IConfidentialClientApplication _clientApp;
    private readonly string[] _scopes;
    private GraphServiceClient _graphClient;

    public TeamsMeetingHelper()
    {
        // Set your Azure app credentials
        string clientId = "YOUR_CLIENT_ID";
        string tenantId = "YOUR_TENANT_ID";
        string clientSecret = "YOUR_CLIENT_SECRET";

        _clientApp = ConfidentialClientApplicationBuilder.Create(clientId)
            .WithTenantId(tenantId)
            .WithClientSecret(clientSecret)
            .Build();

        _scopes = new[] { "https://graph.microsoft.com/.default" };

        // Initialize Graph client
        _graphClient = new GraphServiceClient(this);
    }

    // Authenticates each request
    public async Task AuthenticateRequestAsync(RequestInformation request)
    {
        var result = await _clientApp.AcquireTokenForClient(_scopes).ExecuteAsync();
        request.Headers.Add("Authorization", $"Bearer {result.AccessToken}");
    }

    public Task AuthenticateRequestAsync(RequestInformation request, Dictionary<string, object> additionalAuthenticationContext = null, CancellationToken cancellationToken = default)
    {
        throw new NotImplementedException();
    }

    // Generates the meeting link
    public async Task<string> CreateMeeting(string subject, DateTime start, DateTime end)
    {
        try
        {
            string organizerEmail = "organizer@yourdomain.com"; // Replace with actual organizer

            var onlineMeeting = new OnlineMeeting
            {
                Subject = subject,
                StartDateTime = start,
                EndDateTime = end
            };

            var createdMeeting = await _graphClient.Users[organizerEmail]
                                                   .OnlineMeetings
                                                   .PostAsync(onlineMeeting);

            return createdMeeting.JoinWebUrl;
        }
        catch (Exception ex)
        {
            return "ERROR: " + ex.Message;
        }
    }
}
