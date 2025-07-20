namespace LifeinnovirorMentalHealthConsultency.Migrations
{
    using System;
    using System.Data.Entity.Migrations;
    
    public partial class initializeDatabase : DbMigration
    {
        public override void Up()
        {
            CreateTable(
                "dbo.Admins",
                c => new
                    {
                        AdminId = c.Int(nullable: false, identity: true),
                        Username = c.String(nullable: false, maxLength: 50),
                        PasswordHash = c.String(nullable: false, maxLength: 255),
                        FullName = c.String(nullable: false, maxLength: 100),
                        Email = c.String(nullable: false, maxLength: 100),
                        CreatedAt = c.DateTime(nullable: false),
                    })
                .PrimaryKey(t => t.AdminId);
            
            CreateTable(
                "dbo.Appointments",
                c => new
                    {
                        AppointmentId = c.Int(nullable: false, identity: true),
                        PatientId = c.Int(nullable: false),
                        DoctorId = c.Int(nullable: false),
                        SlotId = c.Int(nullable: false),
                        AppointmentTypeId = c.Int(nullable: false),
                        MeetingMedium = c.String(nullable: false, maxLength: 20),
                        MeetingLink = c.String(),
                        PaymentId = c.Int(),
                        Status = c.String(nullable: false, maxLength: 20),
                        Notes = c.String(),
                        BookedAt = c.DateTime(nullable: false),
                        CancelledAt = c.DateTime(),
                        ConfirmedAt = c.DateTime(),
                    })
                .PrimaryKey(t => t.AppointmentId)
                .ForeignKey("dbo.AppointmentTypes", t => t.AppointmentTypeId, cascadeDelete: true)
                .ForeignKey("dbo.Doctors", t => t.DoctorId, cascadeDelete: true)
                .ForeignKey("dbo.Patients", t => t.PatientId, cascadeDelete: true)
                .ForeignKey("dbo.DoctorTimeSlots", t => t.SlotId, cascadeDelete: true)
                .Index(t => t.PatientId)
                .Index(t => t.DoctorId)
                .Index(t => t.SlotId)
                .Index(t => t.AppointmentTypeId);
            
            CreateTable(
                "dbo.AppointmentTypes",
                c => new
                    {
                        AppointmentTypeId = c.Int(nullable: false, identity: true),
                        Name = c.String(nullable: false, maxLength: 50),
                        Cost = c.Decimal(nullable: false, precision: 18, scale: 2),
                    })
                .PrimaryKey(t => t.AppointmentTypeId);
            
            CreateTable(
                "dbo.Doctors",
                c => new
                    {
                        DoctorId = c.Int(nullable: false, identity: true),
                        FullName = c.String(nullable: false, maxLength: 100),
                        Email = c.String(nullable: false, maxLength: 100),
                        PhoneNumber = c.String(nullable: false, maxLength: 20),
                        PasswordHash = c.String(nullable: false, maxLength: 255),
                        SecurityQuestion = c.String(nullable: false),
                        SecurityAnswerHash = c.String(nullable: false),
                        Qualifications = c.String(),
                        ExperienceSummary = c.String(),
                        YearsOfExperience = c.Int(),
                        minimumCancelTime = c.Int(nullable: false),
                        ProfilePhotoUrl = c.String(maxLength: 255),
                        Status = c.String(nullable: false, maxLength: 20),
                        CreatedAt = c.DateTime(nullable: false),
                        UpdatedAt = c.DateTime(nullable: false),
                    })
                .PrimaryKey(t => t.DoctorId);
            
            CreateTable(
                "dbo.DoctorTimeSlots",
                c => new
                    {
                        SlotId = c.Int(nullable: false, identity: true),
                        DoctorId = c.Int(nullable: false),
                        Date = c.DateTime(nullable: false),
                        StartTime = c.Time(nullable: false, precision: 7),
                        EndTime = c.Time(nullable: false, precision: 7),
                        IsBooked = c.Boolean(nullable: false),
                    })
                .PrimaryKey(t => t.SlotId)
                .ForeignKey("dbo.Doctors", t => t.DoctorId)
                .Index(t => t.DoctorId);
            
            CreateTable(
                "dbo.Patients",
                c => new
                    {
                        PatientId = c.Int(nullable: false, identity: true),
                        FullName = c.String(nullable: false, maxLength: 100),
                        DateOfBirth = c.DateTime(),
                        Gender = c.String(),
                        Email = c.String(),
                        PhoneNumber = c.String(maxLength: 20),
                        Address = c.String(),
                        EmergencyNumber = c.String(maxLength: 20),
                        MedicalHistoryText = c.String(),
                        Height = c.Decimal(precision: 18, scale: 2),
                        Weight = c.Decimal(precision: 18, scale: 2),
                        Religion = c.String(maxLength: 50),
                        EducationDetails = c.String(),
                        Allergies = c.String(),
                        SkinTone = c.String(maxLength: 50),
                        ProfilePhotoUrl = c.String(maxLength: 255),
                        CreatedAt = c.DateTime(nullable: false),
                        UpdatedAt = c.DateTime(nullable: false),
                    })
                .PrimaryKey(t => t.PatientId);
            
            CreateTable(
                "dbo.Certificates",
                c => new
                    {
                        CertificateId = c.Int(nullable: false, identity: true),
                        DoctorId = c.Int(nullable: false),
                        Url = c.String(nullable: false, maxLength: 255),
                        Description = c.String(maxLength: 255),
                    })
                .PrimaryKey(t => t.CertificateId)
                .ForeignKey("dbo.Doctors", t => t.DoctorId, cascadeDelete: true)
                .Index(t => t.DoctorId);
            
            CreateTable(
                "dbo.DoctorLanguages",
                c => new
                    {
                        DoctorId = c.Int(nullable: false),
                        LanguageId = c.Int(nullable: false),
                    })
                .PrimaryKey(t => new { t.DoctorId, t.LanguageId })
                .ForeignKey("dbo.Doctors", t => t.DoctorId, cascadeDelete: true)
                .ForeignKey("dbo.Languages", t => t.LanguageId, cascadeDelete: true)
                .Index(t => t.DoctorId)
                .Index(t => t.LanguageId);
            
            CreateTable(
                "dbo.Languages",
                c => new
                    {
                        LanguageId = c.Int(nullable: false, identity: true),
                        Name = c.String(nullable: false, maxLength: 50),
                    })
                .PrimaryKey(t => t.LanguageId);
            
            CreateTable(
                "dbo.DoctorStatusHistories",
                c => new
                    {
                        StatusId = c.Int(nullable: false, identity: true),
                        DoctorId = c.Int(nullable: false),
                        Status = c.String(nullable: false, maxLength: 20),
                        ChangedAt = c.DateTime(nullable: false),
                    })
                .PrimaryKey(t => t.StatusId);
            
            CreateTable(
                "dbo.DoctorSubSpecializations",
                c => new
                    {
                        DoctorId = c.Int(nullable: false),
                        SubSpecializationId = c.Int(nullable: false),
                    })
                .PrimaryKey(t => new { t.DoctorId, t.SubSpecializationId })
                .ForeignKey("dbo.Doctors", t => t.DoctorId, cascadeDelete: true)
                .ForeignKey("dbo.SubSpecializations", t => t.SubSpecializationId, cascadeDelete: true)
                .Index(t => t.DoctorId)
                .Index(t => t.SubSpecializationId);
            
            CreateTable(
                "dbo.SubSpecializations",
                c => new
                    {
                        SubSpecializationId = c.Int(nullable: false, identity: true),
                        SpecializationId = c.Int(nullable: false),
                        Name = c.String(nullable: false, maxLength: 100),
                    })
                .PrimaryKey(t => t.SubSpecializationId)
                .ForeignKey("dbo.Specializations", t => t.SpecializationId, cascadeDelete: true)
                .Index(t => t.SpecializationId);
            
            CreateTable(
                "dbo.Specializations",
                c => new
                    {
                        SpecializationId = c.Int(nullable: false, identity: true),
                        Name = c.String(nullable: false, maxLength: 100),
                    })
                .PrimaryKey(t => t.SpecializationId);
            
            CreateTable(
                "dbo.FAQs",
                c => new
                    {
                        FaqId = c.Int(nullable: false, identity: true),
                        Question = c.String(nullable: false),
                        Answer = c.String(nullable: false),
                        CreatedAt = c.DateTime(nullable: false),
                        UpdatedAt = c.DateTime(nullable: false),
                    })
                .PrimaryKey(t => t.FaqId);
            
            CreateTable(
                "dbo.Feedbacks",
                c => new
                    {
                        FeedbackId = c.Int(nullable: false, identity: true),
                        AppointmentId = c.Int(nullable: false),
                        PatientId = c.Int(nullable: false),
                        Rating = c.Int(nullable: false),
                        Comments = c.String(),
                        SubmittedAt = c.DateTime(nullable: false),
                    })
                .PrimaryKey(t => t.FeedbackId);
            
            CreateTable(
                "dbo.LeaveRequests",
                c => new
                    {
                        LeaveId = c.Int(nullable: false, identity: true),
                        DoctorId = c.Int(nullable: false),
                        StartDate = c.DateTime(nullable: false),
                        EndDate = c.DateTime(nullable: false),
                        Reason = c.String(),
                        Status = c.String(nullable: false, maxLength: 20),
                        RequestedAt = c.DateTime(nullable: false),
                        ReviewedAt = c.DateTime(),
                        AdminId = c.Int(),
                    })
                .PrimaryKey(t => t.LeaveId);
            
            CreateTable(
                "dbo.Notifications",
                c => new
                    {
                        NotificationId = c.Int(nullable: false, identity: true),
                        RecipientType = c.String(nullable: false),
                        RecipientId = c.Int(nullable: false),
                        Message = c.String(nullable: false),
                        SentAt = c.DateTime(nullable: false),
                        Read = c.Boolean(nullable: false),
                    })
                .PrimaryKey(t => t.NotificationId);
            
            CreateTable(
                "dbo.Payments",
                c => new
                    {
                        PaymentId = c.Int(nullable: false, identity: true),
                        AppointmentId = c.Int(nullable: false),
                        Amount = c.Decimal(nullable: false, precision: 18, scale: 2),
                        PaidAt = c.DateTime(nullable: false),
                        Method = c.String(),
                        TransactionId = c.String(),
                        Status = c.String(nullable: false, maxLength: 20),
                    })
                .PrimaryKey(t => t.PaymentId);
            
            CreateTable(
                "dbo.RegistrationRequests",
                c => new
                    {
                        RequestId = c.Int(nullable: false, identity: true),
                        DoctorId = c.Int(nullable: false),
                        Status = c.String(nullable: false, maxLength: 20),
                        SubmittedAt = c.DateTime(nullable: false),
                        ReviewedAt = c.DateTime(),
                        AdminId = c.Int(),
                        Notes = c.String(),
                    })
                .PrimaryKey(t => t.RequestId)
                .ForeignKey("dbo.Admins", t => t.AdminId)
                .ForeignKey("dbo.Doctors", t => t.DoctorId, cascadeDelete: true)
                .Index(t => t.DoctorId)
                .Index(t => t.AdminId);
            
            CreateTable(
                "dbo.Reports",
                c => new
                    {
                        ReportId = c.Int(nullable: false, identity: true),
                        AppointmentId = c.Int(nullable: false),
                        DoctorId = c.Int(nullable: false),
                        UploadUrl = c.String(nullable: false),
                        UploadedAt = c.DateTime(nullable: false),
                    })
                .PrimaryKey(t => t.ReportId);
            
            CreateTable(
                "dbo.SecureChatMessages",
                c => new
                    {
                        MessageId = c.Int(nullable: false, identity: true),
                        FromUserId = c.Int(nullable: false),
                        ToUserId = c.Int(nullable: false),
                        SentAt = c.DateTime(nullable: false),
                        Content = c.String(nullable: false),
                    })
                .PrimaryKey(t => t.MessageId);
            
            CreateTable(
                "dbo.SystemLogs",
                c => new
                    {
                        LogId = c.Int(nullable: false, identity: true),
                        ActorType = c.String(nullable: false),
                        ActorId = c.Int(nullable: false),
                        Action = c.String(nullable: false),
                        Details = c.String(),
                        CreatedAt = c.DateTime(nullable: false),
                    })
                .PrimaryKey(t => t.LogId);
            
        }
        
        public override void Down()
        {
            DropForeignKey("dbo.RegistrationRequests", "DoctorId", "dbo.Doctors");
            DropForeignKey("dbo.RegistrationRequests", "AdminId", "dbo.Admins");
            DropForeignKey("dbo.DoctorSubSpecializations", "SubSpecializationId", "dbo.SubSpecializations");
            DropForeignKey("dbo.SubSpecializations", "SpecializationId", "dbo.Specializations");
            DropForeignKey("dbo.DoctorSubSpecializations", "DoctorId", "dbo.Doctors");
            DropForeignKey("dbo.DoctorLanguages", "LanguageId", "dbo.Languages");
            DropForeignKey("dbo.DoctorLanguages", "DoctorId", "dbo.Doctors");
            DropForeignKey("dbo.Certificates", "DoctorId", "dbo.Doctors");
            DropForeignKey("dbo.Appointments", "SlotId", "dbo.DoctorTimeSlots");
            DropForeignKey("dbo.Appointments", "PatientId", "dbo.Patients");
            DropForeignKey("dbo.Appointments", "DoctorId", "dbo.Doctors");
            DropForeignKey("dbo.DoctorTimeSlots", "DoctorId", "dbo.Doctors");
            DropForeignKey("dbo.Appointments", "AppointmentTypeId", "dbo.AppointmentTypes");
            DropIndex("dbo.RegistrationRequests", new[] { "AdminId" });
            DropIndex("dbo.RegistrationRequests", new[] { "DoctorId" });
            DropIndex("dbo.SubSpecializations", new[] { "SpecializationId" });
            DropIndex("dbo.DoctorSubSpecializations", new[] { "SubSpecializationId" });
            DropIndex("dbo.DoctorSubSpecializations", new[] { "DoctorId" });
            DropIndex("dbo.DoctorLanguages", new[] { "LanguageId" });
            DropIndex("dbo.DoctorLanguages", new[] { "DoctorId" });
            DropIndex("dbo.Certificates", new[] { "DoctorId" });
            DropIndex("dbo.DoctorTimeSlots", new[] { "DoctorId" });
            DropIndex("dbo.Appointments", new[] { "AppointmentTypeId" });
            DropIndex("dbo.Appointments", new[] { "SlotId" });
            DropIndex("dbo.Appointments", new[] { "DoctorId" });
            DropIndex("dbo.Appointments", new[] { "PatientId" });
            DropTable("dbo.SystemLogs");
            DropTable("dbo.SecureChatMessages");
            DropTable("dbo.Reports");
            DropTable("dbo.RegistrationRequests");
            DropTable("dbo.Payments");
            DropTable("dbo.Notifications");
            DropTable("dbo.LeaveRequests");
            DropTable("dbo.Feedbacks");
            DropTable("dbo.FAQs");
            DropTable("dbo.Specializations");
            DropTable("dbo.SubSpecializations");
            DropTable("dbo.DoctorSubSpecializations");
            DropTable("dbo.DoctorStatusHistories");
            DropTable("dbo.Languages");
            DropTable("dbo.DoctorLanguages");
            DropTable("dbo.Certificates");
            DropTable("dbo.Patients");
            DropTable("dbo.DoctorTimeSlots");
            DropTable("dbo.Doctors");
            DropTable("dbo.AppointmentTypes");
            DropTable("dbo.Appointments");
            DropTable("dbo.Admins");
        }
    }
}
