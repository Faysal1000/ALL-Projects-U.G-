namespace LifeinnovirorMentalHealthConsultency.Migrations
{
    using System;
    using System.Data.Entity.Migrations;
    
    public partial class patientTableChanged : DbMigration
    {
        public override void Up()
        {
            AlterColumn("dbo.Appointments", "Status", c => c.String(maxLength: 20));
            AlterColumn("dbo.Patients", "Email", c => c.String(nullable: false));
            AlterColumn("dbo.Patients", "PasswordHash", c => c.String(nullable: false, maxLength: 255));
        }
        
        public override void Down()
        {
            AlterColumn("dbo.Patients", "PasswordHash", c => c.String(maxLength: 255));
            AlterColumn("dbo.Patients", "Email", c => c.String());
            AlterColumn("dbo.Appointments", "Status", c => c.String(nullable: false, maxLength: 20));
        }
    }
}
